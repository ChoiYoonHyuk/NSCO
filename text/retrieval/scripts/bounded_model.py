"""Raw-E5-anchored exact paths with a bounded, shared residual relation.

For normalized x,y the score is 20<x,y> + c*tanh(6< Ux,Vy >).
U uses the standard bias-free Linear initializer and V starts at zero.
The same U,V score the first and second hop. At initialization every score
and path probability equals the frozen E5 control. The residual changes
each score by at most c; this does not guarantee retained task accuracy.

Per-hop log-probability changes are bounded by 2c, hence ordered path and
terminal marginal log-probability changes by 4c. These worst-case bounds
can still be loose. No extra KL penalty, gold inference prefix, or encoder
update is introduced. Objectives are imported from the frozen baseline module.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from frozen_model import CONDITIONS, loss_terms


def path_probabilities(first_scores, transition_scores, mask):
    """Backward-safe exact marginalization over all valid nonself paths."""
    width = mask.shape[1]
    pair_mask = (mask[:, :, None] & mask[:, None, :] &
                 ~torch.eye(width, dtype=torch.bool, device=mask.device)[None])
    first = F.log_softmax(first_scores.masked_fill(~mask, -torch.inf), dim=-1)
    scores = transition_scores.masked_fill(~pair_mask, -torch.inf)
    safe_scores = scores.masked_fill(~mask[:, :, None], 0.0)
    transition = F.log_softmax(safe_scores, dim=-1).masked_fill(~pair_mask, -torch.inf)
    joint = first[:, :, None] + transition
    safe_joint = joint.masked_fill(~mask[:, None, :], 0.0)
    terminal = torch.logsumexp(safe_joint, dim=1).masked_fill(~mask, -torch.inf)
    return first, transition, joint, terminal, pair_mask


class AnchoredTwoHopRetriever(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, kind="chain", logit_scale=20.0,
                 residual_cap=1.0, residual_scale=6.0):
        super().__init__()
        if kind not in ("direct", "chain") or min(input_dim, projection_dim) <= 0:
            raise ValueError("Invalid architecture or projection dimensions")
        if not all(math.isfinite(value) and value > 0 for value in
                   (logit_scale, residual_cap, residual_scale)):
            raise ValueError("Fixed scales and residual cap must be positive and finite")
        self.input_dim, self.projection_dim = input_dim, projection_dim
        self.kind, self.logit_scale = kind, float(logit_scale)
        self.residual_cap, self.residual_scale = float(residual_cap), float(residual_scale)
        self.query_adapter = nn.Linear(input_dim, projection_dim, bias=False)
        self.key_adapter = nn.Linear(input_dim, projection_dim, bias=False)
        nn.init.zeros_(self.key_adapter.weight)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "query_adapter": self.query_adapter.weight.numel(),
                "key_adapter": self.key_adapter.weight.numel()}

    def forward(self, q: Tensor, p: Tensor, candidate_mask: Tensor,
                context_queries: Tensor | None = None) -> dict[str, Tensor]:
        if q.ndim != 2 or p.ndim != 3 or candidate_mask.ndim != 2:
            raise ValueError("Expected q[B,D], p[B,N,D], mask[B,N]")
        if (q.shape[0] != p.shape[0] or q.shape[-1] != self.input_dim or
                p.shape[-1] != self.input_dim or candidate_mask.shape != p.shape[:2]):
            raise ValueError("Mismatched input dimensions")
        if candidate_mask.dtype != torch.bool or not (q.device == p.device == candidate_mask.device):
            raise ValueError("Boolean mask and inputs must share a device")
        if not bool((candidate_mask.sum(dim=1) >= 2).all()):
            raise ValueError("Every example needs at least two valid candidates")
        if context_queries is not None:
            if context_queries.shape != p.shape or context_queries.device != p.device:
                raise ValueError("Context queries must have paragraph shape/device")
            context_queries = context_queries.masked_fill(~candidate_mask[..., None], 0.0)
        p = p.masked_fill(~candidate_mask[..., None], 0.0)
        # Match the identity-projection raw baseline's normalization exactly.
        query = F.normalize(q, dim=-1)
        keys = F.normalize(p, dim=-1)
        sources = F.normalize(p if context_queries is None else context_queries, dim=-1)
        raw_first = self.logit_scale * torch.einsum("bnd,bd->bn", keys, query)
        raw_transition = self.logit_scale * torch.einsum("bid,bjd->bij", sources, keys)
        adapted_keys = self.key_adapter(keys)
        first_interaction = torch.einsum("bnd,bd->bn", adapted_keys, self.query_adapter(query))
        transition_interaction = torch.einsum("bid,bjd->bij", self.query_adapter(sources), adapted_keys)
        first_delta = self.residual_cap * torch.tanh(self.residual_scale * first_interaction)
        transition_delta = self.residual_cap * torch.tanh(self.residual_scale * transition_interaction)
        first_scores, transition_scores = raw_first + first_delta, raw_transition + transition_delta
        first, transition, joint, terminal, pair_mask = path_probabilities(first_scores, transition_scores, candidate_mask)
        width = p.shape[1]
        flat = joint.flatten(1).argmax(dim=1)
        joint_map = torch.stack((flat // width, flat % width), dim=1)
        terminal_index = terminal.argmax(dim=1)
        first_index = joint.gather(2, terminal_index[:, None, None].expand(-1, width, 1)).squeeze(-1).argmax(dim=1)
        marginal_path = torch.stack((first_index, terminal_index), dim=1)
        unordered = torch.logaddexp(joint, joint.transpose(1, 2))
        upper = torch.triu(torch.ones((width, width), dtype=torch.bool, device=q.device), diagonal=1)
        unordered = unordered.masked_fill(~(pair_mask & upper[None]), -torch.inf)
        unordered_flat = unordered.flatten(1).argmax(dim=1)
        unordered_pair = torch.stack((unordered_flat // width, unordered_flat % width), dim=1)
        direct_logits = first_scores.masked_fill(~candidate_mask, -torch.inf)
        selected_terminal = first if self.kind == "direct" else terminal
        return {"first_logits": first, "terminal_logits": selected_terminal,
                "first_log_probs": first, "terminal_log_probs": selected_terminal,
                "chain_terminal_log_probs": terminal, "direct_logits": direct_logits,
                "transition_log_probs": transition, "joint_log_probs": joint,
                "joint_map_indices": joint_map, "marginal_terminal_path_indices": marginal_path,
                "unordered_pair_indices": unordered_pair, "direct_top2_indices": direct_logits.topk(2, dim=1).indices,
                "raw_first_scores": raw_first.masked_fill(~candidate_mask, -torch.inf),
                "raw_transition_scores": raw_transition.masked_fill(~pair_mask, -torch.inf),
                "first_score_perturbation": first_delta, "transition_score_perturbation": transition_delta}


def build_model(kind="chain", input_dim=768, projection_dim=128, logit_scale=20.0,
                residual_cap=1.0, residual_scale=6.0):
    if kind in CONDITIONS:
        kind = "direct" if kind == "direct_terminal" else "chain"
    return AnchoredTwoHopRetriever(input_dim, projection_dim, kind, logit_scale,
                                   residual_cap, residual_scale)


def self_test():
    """Synthetic CPU checks only. Does not read any benchmark or GPU outcome."""
    from frozen_model import build_model as build_raw
    torch.set_num_threads(2)
    torch.manual_seed(718202)
    q = F.normalize(torch.randn(3, 10, dtype=torch.float64), dim=-1)
    p = F.normalize(torch.randn(3, 5, 10, dtype=torch.float64), dim=-1)
    contexts = F.normalize(q[:, None, :] + .7 * p, dim=-1)
    mask = torch.arange(5)[None, :] < torch.tensor([5, 4, 2])[:, None]
    first_target, terminal_target = torch.tensor([0, 0, 1]), torch.tensor([3, 2, 0])
    checks = []
    for cap in (1.0, 4.0):
        model = build_model(input_dim=10, projection_dim=7, residual_cap=cap).double()
        raw = build_raw(input_dim=10, projection_dim=10).double()
        with torch.no_grad():
            raw.query_projection.weight.copy_(torch.eye(10, dtype=q.dtype))
            raw.key_projection.weight.copy_(torch.eye(10, dtype=q.dtype))
        assert bool((model.key_adapter.weight == 0).all())
        assert float(model.query_adapter.weight.norm()) > 0
        for source in (None, contexts, q[:, None, :].expand_as(p)):
            initial, baseline = model(q, p, mask, source), raw(q, p, mask, source)
            for key in ("first_logits", "terminal_logits", "transition_log_probs", "joint_log_probs"):
                torch.testing.assert_close(initial[key], baseline[key], atol=0, rtol=0)
        output = model(q, p, mask, contexts)
        loss_terms(output, terminal_target, first_target, condition="chain_first100")["loss"].backward()
        torch.testing.assert_close(model.query_adapter.weight.grad,
                                   torch.zeros_like(model.query_adapter.weight), atol=0, rtol=0)
        assert bool(torch.isfinite(model.key_adapter.weight.grad).all()) and float(model.key_adapter.weight.grad.norm()) > 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        loss_terms(model(q, p, mask, contexts), terminal_target, first_target, condition="chain_first100")["loss"].backward()
        assert bool(torch.isfinite(model.query_adapter.weight.grad).all()) and float(model.query_adapter.weight.grad.norm()) > 0
        trained_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        # Extreme adapters exercise the actual bound without assuming a small fit.
        with torch.no_grad():
            model.query_adapter.weight.normal_(std=2)
            model.key_adapter.weight.normal_(std=2)
        changed = model(q, p, mask, contexts)
        baseline = raw(q, p, mask, contexts)
        pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(5, dtype=torch.bool)[None]
        assert float(changed["first_score_perturbation"].abs().max()) <= cap
        assert float(changed["transition_score_perturbation"].abs().max()) <= cap
        assert float((changed["first_logits"] - baseline["first_logits"])[mask].abs().max()) <= 2 * cap + 1e-12
        assert float((changed["joint_log_probs"] - baseline["joint_log_probs"])[pair].abs().max()) <= 4 * cap + 1e-12
        assert float((changed["terminal_logits"] - baseline["terminal_logits"])[mask].abs().max()) <= 4 * cap + 1e-12
        # Manual three-candidate probability-space evaluation with nonzero V.
        sample_p, sample_c, sample_mask = p[1:2, :3], contexts[1:2, :3], torch.ones(1, 3, dtype=torch.bool)
        result = model(q[1:2], sample_p, sample_mask, sample_c)
        x, y, c = F.normalize(q[1], dim=-1), F.normalize(sample_p[0], dim=-1), F.normalize(sample_c[0], dim=-1)
        def score(left, right):
            return 20 * float(torch.dot(left, right)) + cap * math.tanh(6 * float(torch.dot(model.query_adapter(left), model.key_adapter(right))))
        def softmax(values):
            weights = [math.exp(value - max(values)) for value in values]
            return [weight / sum(weights) for weight in weights]
        one = softmax([score(x, key) for key in y])
        two = [softmax([score(c[i], y[j]) if i != j else -math.inf for j in range(3)]) for i in range(3)]
        joint = torch.tensor([[one[i] * two[i][j] for j in range(3)] for i in range(3)], dtype=q.dtype)
        torch.testing.assert_close(result["joint_log_probs"][0].exp(), joint, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(result["terminal_logits"][0].exp(), joint.sum(0), atol=1e-12, rtol=1e-10)
        permutations = torch.stack([torch.randperm(5) for _ in range(3)])
        inverse = permutations.argsort(1)
        shuffled = model(q, p.gather(1, permutations[..., None].expand_as(p)), mask.gather(1, permutations),
                         contexts.gather(1, permutations[..., None].expand_as(contexts)))
        dirty_p, dirty_c = p.clone(), contexts.clone()
        dirty_p[~mask] = torch.nan; dirty_c[~mask] = torch.nan
        poisoned = model(q, dirty_p, mask, dirty_c)
        for key in ("first_logits", "terminal_logits"):
            torch.testing.assert_close(changed[key], shuffled[key].gather(1, inverse), atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(changed[key], poisoned[key], atol=0, rtol=0)
        model.load_state_dict(trained_state)
        model.zero_grad(set_to_none=True)
        loss_terms(model(q, p, mask, contexts), terminal_target, first_target, condition="chain_first100")["loss"].backward()
        derivatives = {}
        for name, parameter in model.named_parameters():
            gradient = parameter.grad.detach().clone(); direction = gradient / gradient.norm()
            origin = parameter.detach().clone(); epsilon = 1e-5
            with torch.no_grad():
                parameter.copy_(origin + epsilon * direction)
                plus = loss_terms(model(q, p, mask, contexts), terminal_target, first_target, condition="chain_first100")["loss"].item()
                parameter.copy_(origin - epsilon * direction)
                minus = loss_terms(model(q, p, mask, contexts), terminal_target, first_target, condition="chain_first100")["loss"].item()
                parameter.copy_(origin)
            finite = (plus - minus) / (2 * epsilon); analytic = float((gradient * direction).sum())
            assert math.isclose(finite, analytic, rel_tol=1e-4, abs_tol=1e-6)
            derivatives[name] = {"analytic": analytic, "finite_difference": finite}
        for condition in CONDITIONS:
            model.zero_grad(set_to_none=True)
            losses = loss_terms(model(q, p, mask, contexts), terminal_target, first_target, condition=condition,
                                random_target=torch.tensor([2, 1, 0]))
            losses["loss"].backward()
            assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        checks.append({"residual_cap": cap, "exact_initial_raw_control": True,
                       "initial_U_gradient_zero_V_gradient_nonzero": True, "second_U_gradient_nonzero": True,
                       "score_and_log_probability_bounds": True, "manual_N3_probabilities": True,
                       "permutation_and_NaN_padding": True, "all_objective_gradients_finite": True,
                       "directional_derivatives": derivatives})
    return {"status": "passed", "device": "cpu", "benchmark_outcomes_read": False,
            "checks": checks, "parameters_at_768x128": 2 * 768 * 128,
            "interpretation": "Anchored scores and step-zero fallback do not guarantee retained accuracy after fitting or on another dataset."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
