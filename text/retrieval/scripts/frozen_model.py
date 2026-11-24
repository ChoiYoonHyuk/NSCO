"""Exact two-hop latent-path retrieval with a shared asymmetric relation.

The same query projection A and key projection B score q->p_i and p_i->p_j.
Terminal training marginalizes every valid first paragraph; no gold prefix,
gold decomposition text, or answer enters forward. Self-transitions are
excluded identically for every chain condition. A direct comparator uses
the same parameters and its question->paragraph distribution as terminal.
Projections start equal to one another, without a frozen-cosine residual.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CONDITIONS = ("direct_terminal", "chain_terminal", "chain_first010", "chain_first100",
              "chain_random100", "chain_teacher", "chain_first", "chain_random")
AUXILIARY_COEFFICIENTS = {"chain_first010": 0.1, "chain_first100": 1.0,
                          "chain_random100": 1.0, "chain_teacher": 1.0,
                          "chain_first": 1.0, "chain_random": 1.0}


class ExactTwoHopRetriever(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, kind="chain", logit_scale=20.0):
        super().__init__()
        if kind not in ("chain", "direct"):
            raise ValueError("kind must be chain or direct")
        if min(input_dim, projection_dim) <= 0:
            raise ValueError("Feature/projection dimensions must be positive")
        if not math.isfinite(logit_scale) or not 0 < logit_scale < 1000:
            raise ValueError("Invalid fixed logit scale")
        self.input_dim, self.projection_dim = input_dim, projection_dim
        self.kind, self.logit_scale = kind, logit_scale
        self.query_projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.key_projection = nn.Linear(input_dim, projection_dim, bias=False)
        with torch.no_grad():
            self.key_projection.weight.copy_(self.query_projection.weight)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "query_projection": self.query_projection.weight.numel(),
                "key_projection": self.key_projection.weight.numel()}

    def forward(self, q: Tensor, p: Tensor, candidate_mask: Tensor,
                context_queries: Tensor | None = None) -> dict[str, Tensor]:
        if q.ndim != 2 or p.ndim != 3 or candidate_mask.ndim != 2:
            raise ValueError("Expected q[B,D], p[B,N,D], mask[B,N]")
        if (q.shape[0] != p.shape[0] or q.shape[-1] != self.input_dim or
                p.shape[-1] != self.input_dim or candidate_mask.shape != p.shape[:2]):
            raise ValueError("Mismatched input dimensions")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("Candidate mask must be boolean")
        if not (q.device == p.device == candidate_mask.device):
            raise ValueError("All inputs must share a device")
        if not bool((candidate_mask.sum(dim=1) >= 2).all()):
            raise ValueError("Every two-hop example needs at least two valid candidates")
        if context_queries is not None:
            if context_queries.shape != p.shape or context_queries.device != p.device:
                raise ValueError("Context queries must have paragraph shape/device")
            context_queries = context_queries.masked_fill(~candidate_mask[..., None], 0.0)
        p = p.masked_fill(~candidate_mask[..., None], 0.0)
        initial_query = F.normalize(self.query_projection(q), dim=-1)
        paragraph_queries = F.normalize(self.query_projection(
            p if context_queries is None else context_queries), dim=-1)
        paragraph_keys = F.normalize(self.key_projection(p), dim=-1)
        first_scores = self.logit_scale * torch.einsum("bnd,bd->bn", paragraph_keys, initial_query)
        first_scores = first_scores.masked_fill(~candidate_mask, -torch.inf)
        log_first = F.log_softmax(first_scores, dim=-1)
        transition_scores = self.logit_scale * torch.einsum("bid,bjd->bij", paragraph_queries, paragraph_keys)
        n = p.shape[1]
        pair_mask = (candidate_mask[:, :, None] & candidate_mask[:, None, :] &
                     ~torch.eye(n, dtype=torch.bool, device=p.device)[None])
        transition_scores = transition_scores.masked_fill(~pair_mask, -torch.inf)
        # Invalid source rows otherwise have log_softmax(-inf,...,-inf), whose
        # NaN backward can contaminate valid gradients even after masking.
        safe_transition_scores = transition_scores.masked_fill(~candidate_mask[:, :, None], 0.0)
        log_transition = F.log_softmax(safe_transition_scores, dim=-1).masked_fill(~pair_mask, -torch.inf)
        log_joint = log_first[:, :, None] + log_transition
        # Likewise avoid all--inf logsumexp for an invalid terminal column.
        safe_joint = log_joint.masked_fill(~candidate_mask[:, None, :], 0.0)
        log_terminal = torch.logsumexp(safe_joint, dim=1).masked_fill(~candidate_mask, -torch.inf)
        flat_path = log_joint.flatten(1).argmax(dim=1)
        joint_map = torch.stack((flat_path // n, flat_path % n), dim=1)
        marginal_terminal = log_terminal.argmax(dim=1)
        conditional_first = log_joint.gather(2, marginal_terminal[:, None, None].expand(-1, n, 1)).squeeze(-1).argmax(dim=1)
        marginal_path = torch.stack((conditional_first, marginal_terminal), dim=1)
        # An unordered pair has probability mass for both possible directions.
        # Use one triangular copy for decoding; this is distinct from joint MAP.
        unordered_scores = torch.logaddexp(log_joint, log_joint.transpose(1, 2))
        upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=p.device), diagonal=1)
        unordered_scores = unordered_scores.masked_fill(~(pair_mask & upper[None]), -torch.inf)
        unordered_flat = unordered_scores.flatten(1).argmax(dim=1)
        unordered_pair = torch.stack((unordered_flat // n, unordered_flat % n), dim=1)
        terminal_output = log_first if self.kind == "direct" else log_terminal
        return {"first_logits": log_first,
                "terminal_logits": terminal_output,
                "first_log_probs": log_first, "terminal_log_probs": terminal_output,
                "chain_terminal_log_probs": log_terminal,
                "direct_logits": first_scores,
                "transition_log_probs": log_transition, "joint_log_probs": log_joint,
                "joint_map_indices": joint_map, "marginal_terminal_path_indices": marginal_path,
                "unordered_pair_indices": unordered_pair,
                "direct_top2_indices": first_scores.topk(2, dim=1).indices}


def build_model(kind="chain", input_dim=768, projection_dim=128, logit_scale=20.0):
    if kind in CONDITIONS:
        kind = "direct" if kind == "direct_terminal" else "chain"
    return ExactTwoHopRetriever(input_dim, projection_dim, kind, logit_scale)


def _check_target(log_probs, target, name):
    if target is None or target.dtype != torch.long or target.shape != log_probs.shape[:1]:
        raise ValueError(f"{name} must be a length-B long tensor")
    if target.device != log_probs.device:
        raise ValueError(f"{name} and scores must share a device")
    if not bool(((target >= 0) & (target < log_probs.shape[1])).all()):
        raise ValueError(f"{name} is outside the candidate range")
    if not bool(torch.isfinite(log_probs.gather(1, target[:, None])).all()):
        raise ValueError(f"{name} selects padding or a nonfinite score")


def loss_terms(outputs, terminal_target, first_target=None, condition="chain_terminal",
               random_target=None, auxiliary_weight=1.0):
    if condition not in CONDITIONS:
        raise ValueError("Unknown exact-path condition")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Auxiliary weight must be finite and nonnegative")
    terminal = (outputs["first_log_probs"] if condition == "direct_terminal"
                else outputs["chain_terminal_log_probs"])
    _check_target(terminal, terminal_target, "Terminal target")
    terminal_loss = F.nll_loss(terminal, terminal_target)
    auxiliary_loss = terminal_loss.new_zeros(())
    coefficient = AUXILIARY_COEFFICIENTS.get(condition, 0.0) * auxiliary_weight
    if condition == "chain_teacher":
        _check_target(outputs["first_log_probs"], first_target, "Teacher first target")
        conditional = outputs["transition_log_probs"][torch.arange(len(first_target), device=first_target.device), first_target]
        _check_target(conditional, terminal_target, "Conditional terminal target")
        terminal_loss = F.nll_loss(conditional, terminal_target)
    if coefficient:
        target = random_target if condition in ("chain_random", "chain_random100") else first_target
        _check_target(outputs["first_log_probs"], target, "Auxiliary target")
        auxiliary_loss = F.nll_loss(outputs["first_log_probs"], target)
    return {"loss": terminal_loss + coefficient * auxiliary_loss,
            "terminal_loss": terminal_loss, "auxiliary_loss": auxiliary_loss,
            "effective_auxiliary_weight": terminal_loss.new_tensor(coefficient)}


def self_test():
    """Synthetic CPU validation, including a manually enumerated 3-node graph."""
    torch.set_num_threads(2)
    torch.manual_seed(717202)
    q = F.normalize(torch.randn(3, 8, dtype=torch.float64), dim=-1)
    p = F.normalize(torch.randn(3, 4, 8, dtype=torch.float64), dim=-1)
    mask = torch.arange(4)[None, :] < torch.tensor([4, 3, 2])[:, None]
    model = build_model(input_dim=8, projection_dim=5).double()
    assert torch.equal(model.query_projection.weight, model.key_projection.weight)
    output = model(q, p, mask)
    torch.testing.assert_close(output["first_log_probs"].exp().sum(-1), torch.ones(3, dtype=q.dtype))
    torch.testing.assert_close(output["terminal_log_probs"].exp().sum(-1), torch.ones(3, dtype=q.dtype))
    torch.testing.assert_close(output["transition_log_probs"].exp().sum(-1), mask.to(q.dtype))
    assert bool((output["transition_log_probs"].diagonal(dim1=1, dim2=2).exp() == 0).all())
    # A manual probability-space enumeration, independent of logsumexp.
    item = 1
    a = model.query_projection.weight.detach(); b = model.key_projection.weight.detach()
    query = F.normalize(F.linear(q[item], a), dim=-1)
    origins = F.normalize(F.linear(p[item, :3], a), dim=-1)
    keys = F.normalize(F.linear(p[item, :3], b), dim=-1)
    first_scores = [20.0 * float(torch.dot(query, key)) for key in keys]
    first = [math.exp(value - max(first_scores)) for value in first_scores]
    first = [value / sum(first) for value in first]
    transitions = []
    for i in range(3):
        scores = [20.0 * float(torch.dot(origins[i], keys[j])) if j != i else -math.inf for j in range(3)]
        weights = [math.exp(value - max(scores)) for value in scores]
        transitions.append([value / sum(weights) for value in weights])
    joint = torch.tensor([[first[i] * transitions[i][j] for j in range(3)] for i in range(3)], dtype=q.dtype)
    torch.testing.assert_close(output["joint_log_probs"][item, :3, :3].exp(), joint, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(output["terminal_log_probs"][item, :3].exp(), joint.sum(0), atol=1e-12, rtol=1e-10)
    expected_pair = max(((i, j) for i in range(3) for j in range(i + 1, 3)),
                        key=lambda pair: float(joint[pair[0], pair[1]] + joint[pair[1], pair[0]]))
    assert output["unordered_pair_indices"][item].tolist() == list(expected_pair)
    contexts = F.normalize(q[:, None, :] + 0.7 * p, dim=-1)
    contextual = model(q, p, mask, contexts)
    context_origins = F.normalize(F.linear(contexts[item, :3], a), dim=-1)
    contextual_transitions = []
    for i in range(3):
        scores = [20.0 * float(torch.dot(context_origins[i], keys[j])) if j != i else -math.inf for j in range(3)]
        weights = [math.exp(value - max(scores)) for value in scores]
        contextual_transitions.append([value / sum(weights) for value in weights])
    context_joint = torch.tensor([[first[i] * contextual_transitions[i][j] for j in range(3)] for i in range(3)], dtype=q.dtype)
    torch.testing.assert_close(contextual["terminal_log_probs"][item, :3].exp(), context_joint.sum(0), atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(contextual["first_logits"], output["first_logits"], atol=0, rtol=0)
    dirty = p.clone(); dirty[~mask] = torch.nan
    poisoned = model(q, dirty, mask)
    permutations = torch.stack([torch.randperm(4) for _ in range(3)])
    inverse = permutations.argsort(dim=1)
    permuted = model(q, p.gather(1, permutations[..., None].expand_as(p)), mask.gather(1, permutations))
    for role in ("first_logits", "terminal_logits"):
        torch.testing.assert_close(output[role], poisoned[role], atol=0, rtol=0)
        torch.testing.assert_close(output[role], permuted[role].gather(1, inverse), atol=1e-12, rtol=1e-10)
        assert bool(torch.isneginf(output[role][~mask]).all())
    restored = permuted["transition_log_probs"].gather(1, inverse[:, :, None].expand(-1, -1, 4)).gather(2, inverse[:, None, :].expand(-1, 4, -1))
    torch.testing.assert_close(restored, output["transition_log_probs"], atol=1e-12, rtol=1e-10)
    contextual_permuted = model(q, p.gather(1, permutations[..., None].expand_as(p)),
                                mask.gather(1, permutations), contexts.gather(1, permutations[..., None].expand_as(contexts)))
    dirty_contexts = contexts.clone(); dirty_contexts[~mask] = torch.nan
    contextual_poisoned = model(q, dirty, mask, dirty_contexts)
    for role in ("first_logits", "terminal_logits"):
        torch.testing.assert_close(contextual[role], contextual_permuted[role].gather(1, inverse), atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(contextual[role], contextual_poisoned[role], atol=0, rtol=0)
    terminal_target, first_target, random_target = torch.tensor([2, 1, 0]), torch.tensor([0, 0, 1]), torch.tensor([1, 2, 0])
    for condition in CONDITIONS:
        model.zero_grad(set_to_none=True)
        current = model(q, p, mask)
        losses = loss_terms(current, terminal_target, first_target, condition, random_target)
        selected_terminal = current["first_log_probs"] if condition == "direct_terminal" else current["terminal_log_probs"]
        if condition == "chain_teacher":
            selected_terminal = current["transition_log_probs"][torch.arange(3), first_target]
        manual = F.nll_loss(selected_terminal, terminal_target)
        coefficient = AUXILIARY_COEFFICIENTS.get(condition, 0.0)
        if coefficient:
            auxiliary_target = random_target if condition in ("chain_random", "chain_random100") else first_target
            manual = manual + coefficient * F.nll_loss(current["first_log_probs"], auxiliary_target)
        torch.testing.assert_close(losses["loss"], manual, atol=0, rtol=0)
        losses["loss"].backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.norm()) > 0
    # Both relation matrices receive a terminal-only gradient through exact paths.
    model.zero_grad(set_to_none=True)
    loss_terms(model(q, p, mask), terminal_target)["loss"].backward()
    derivatives = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad.detach().clone(); direction = gradient / gradient.norm()
        origin = parameter.detach().clone(); epsilon = 1e-5
        with torch.no_grad():
            parameter.copy_(origin + epsilon * direction)
            plus = loss_terms(model(q, p, mask), terminal_target)["loss"].item()
            parameter.copy_(origin - epsilon * direction)
            minus = loss_terms(model(q, p, mask), terminal_target)["loss"].item()
            parameter.copy_(origin)
        numerical = (plus - minus) / (2 * epsilon); analytic = float((gradient * direction).sum())
        assert math.isclose(numerical, analytic, rel_tol=1e-5, abs_tol=1e-6)
        derivatives[name] = {"analytic": analytic, "finite_difference": numerical}
    direct = build_model(kind="direct", input_dim=8, projection_dim=5).double()
    direct.load_state_dict(model.state_dict())
    torch.testing.assert_close(direct(q, p, mask)["terminal_logits"], model(q, p, mask)["first_logits"], atol=0, rtol=0)
    return {"status": "passed", "device": "cpu", "benchmark_data_read": False,
            "checks": ["equal_initial_A_B", "normalized_probabilities", "manual_N3_marginal",
                       "manual_unordered_pair", "manual_contextual_N3_marginal",
                       "zero_self_transition", "candidate_permutation", "NaN_padding",
                       "context_query_permutation_and_NaN_padding", "weighted_and_teacher_objectives",
                       "finite_nonzero_A_B_gradients",
                       "terminal_gradient_finite_differences", "direct_parameter_equivalence"],
            "parameters": model.parameter_counts(), "directional_derivatives": derivatives}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
