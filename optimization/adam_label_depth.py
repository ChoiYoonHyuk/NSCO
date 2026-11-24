"""Reproduce equal-label Adam comparisons from archived counts or training outputs."""
import argparse
from collections import Counter
import csv
import io
import os
import random
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

COUNTS = """study,depth,labels,rate,teacher,final,clean,noisy
depth,9,131072,0.003,721301,818,1197,
depth,9,131072,0.003,721313,1043,806,
depth,9,131072,0.003,721327,1269,1418,
depth,9,131072,0.003,721339,790,484,
depth,9,131072,0.003,721351,417,972,
depth,9,131072,0.003,721367,296,498,
depth,9,131072,0.003,721379,354,460,
depth,9,131072,0.003,721393,523,503,
depth,9,131072,0.02,721301,368,649,
depth,9,131072,0.02,721313,406,539,
depth,9,131072,0.02,721327,464,591,
depth,9,131072,0.02,721339,427,630,
depth,9,131072,0.02,721351,2631,579,
depth,9,131072,0.02,721367,356,505,
depth,9,131072,0.02,721379,373,508,
depth,9,131072,0.02,721393,396,721,
depth,12,131072,0.003,721301,4073,1358,
depth,12,131072,0.003,721313,3942,1258,
depth,12,131072,0.003,721327,4092,1546,
depth,12,131072,0.003,721339,4148,952,
depth,12,131072,0.003,721351,4048,1279,
depth,12,131072,0.003,721367,3772,853,
depth,12,131072,0.003,721379,1363,877,
depth,12,131072,0.003,721393,4112,873,
depth,12,131072,0.02,721301,984,722,
depth,12,131072,0.02,721313,494,660,
depth,12,131072,0.02,721327,692,800,
depth,12,131072,0.02,721339,3970,507,
depth,12,131072,0.02,721351,541,838,
depth,12,131072,0.02,721367,807,414,
depth,12,131072,0.02,721379,680,577,
depth,12,131072,0.02,721393,3986,660,
replication,12,131072,0.003,731901,4082,875,834
replication,12,131072,0.003,731902,3986,1141,934
replication,12,131072,0.003,731903,4055,1262,943
replication,12,131072,0.003,731904,3995,549,745
replication,12,131072,0.003,731905,1613,1066,937
replication,12,131072,0.003,731906,3953,1272,1067
replication,12,131072,0.003,731907,4044,942,1204
replication,12,131072,0.003,731908,4078,901,919
replication,12,131072,0.003,731909,2401,811,915
replication,12,131072,0.003,731910,4116,1205,1213
replication,12,131072,0.003,731911,4091,1528,1182
replication,12,131072,0.003,731912,4055,957,978
replication,12,131072,0.003,731913,4179,744,697
replication,12,131072,0.003,731914,1030,879,854
replication,12,131072,0.003,731915,1021,903,821
replication,12,131072,0.003,731916,4043,1603,1351
replication,12,131072,0.02,731901,841,619,677
replication,12,131072,0.02,731902,3997,584,557
replication,12,131072,0.02,731903,854,706,647
replication,12,131072,0.02,731904,4040,554,627
replication,12,131072,0.02,731905,4093,619,803
replication,12,131072,0.02,731906,4009,702,728
replication,12,131072,0.02,731907,4035,701,724
replication,12,131072,0.02,731908,4122,721,688
replication,12,131072,0.02,731909,895,709,632
replication,12,131072,0.02,731910,1712,580,773
replication,12,131072,0.02,731911,877,589,728
replication,12,131072,0.02,731912,855,859,822
replication,12,131072,0.02,731913,849,661,697
replication,12,131072,0.02,731914,640,702,604
replication,12,131072,0.02,731915,2248,542,814
replication,12,131072,0.02,731916,1599,796,573
replication,12,262144,0.003,731901,335,444,481
replication,12,262144,0.003,731902,2457,309,369
replication,12,262144,0.003,731903,786,547,520
replication,12,262144,0.003,731904,4035,280,326
replication,12,262144,0.003,731905,359,507,624
replication,12,262144,0.003,731906,1524,419,479
replication,12,262144,0.003,731907,695,395,465
replication,12,262144,0.003,731908,1180,328,417
replication,12,262144,0.003,731909,326,333,369
replication,12,262144,0.003,731910,4134,493,509
replication,12,262144,0.003,731911,4070,536,525
replication,12,262144,0.003,731912,1878,576,688
replication,12,262144,0.003,731913,4071,487,643
replication,12,262144,0.003,731914,494,275,360
replication,12,262144,0.003,731915,483,363,392
replication,12,262144,0.003,731916,4133,655,891
replication,12,262144,0.02,731901,446,524,556
replication,12,262144,0.02,731902,3298,449,373
replication,12,262144,0.02,731903,645,376,454
replication,12,262144,0.02,731904,617,506,453
replication,12,262144,0.02,731905,576,592,576
replication,12,262144,0.02,731906,539,423,479
replication,12,262144,0.02,731907,372,565,443
replication,12,262144,0.02,731908,2949,442,560
replication,12,262144,0.02,731909,426,522,460
replication,12,262144,0.02,731910,412,443,565
replication,12,262144,0.02,731911,429,600,330
replication,12,262144,0.02,731912,476,539,447
replication,12,262144,0.02,731913,411,398,586
replication,12,262144,0.02,731914,467,445,588
replication,12,262144,0.02,731915,509,433,591
replication,12,262144,0.02,731916,433,384,464
six,6,131072,0.003,711201,267,371,
six,6,131072,0.003,711217,200,274,
six,6,131072,0.003,711229,230,375,
six,6,131072,0.003,711241,161,638,
six,6,131072,0.003,711253,215,393,
six,6,131072,0.02,711201,2687,282,
six,6,131072,0.02,711217,287,340,
six,6,131072,0.02,711229,232,373,
six,6,131072,0.02,711241,314,408,
six,6,131072,0.02,711253,272,444,"""

EXPECTED = {('depth', 9, 131072, 0.003, 'clean'): (1.263427734375, -1.00555419921875, 3.5858154296875),
 ('depth', 9, 131072, 0.02, 'clean'): (-1.06658935546875, -8.11309814453125, 2.8411865234375),
 ('depth', 12, 131072, 0.003, 'clean'): (-31.3629150390625, -36.5570068359375, -23.49700927734375),
 ('depth', 12, 131072, 0.02, 'clean'): (-10.64453125, -24.84283447265625, 0.3753662109375),
 ('replication', 12, 131072, 0.003, 'clean'): (-29.071044921875,
                                               -35.17534255981445,
                                               -22.08709716796875),
 ('replication', 12, 131072, 0.003, 'noisy'): (-29.8675537109375,
                                               -35.8513069152832,
                                               -22.913341522216797),
 ('replication', 12, 131072, 0.02, 'clean'): (-19.09027099609375,
                                              -28.121204376220703,
                                              -10.451488494873045),
 ('replication', 12, 131072, 0.02, 'noisy'): (-18.7469482421875,
                                              -27.64516830444336,
                                              -10.29205322265625),
 ('replication', 12, 262144, 0.003, 'clean'): (-18.320465087890625,
                                               -27.626819610595703,
                                               -9.500789642333979),
 ('replication', 12, 262144, 0.003, 'noisy'): (-17.47283935546875,
                                               -26.724281311035156,
                                               -8.710479736328125),
 ('replication', 12, 262144, 0.02, 'clean'): (-4.0924072265625,
                                              -10.295906066894531,
                                              0.2838134765625),
 ('replication', 12, 262144, 0.02, 'noisy'): (-3.875732421875,
                                              -10.0860595703125,
                                              0.475311279296875),
 ('six', 6, 131072, 0.003, 'clean'): (2.3876953125, 1.22314453125, 4.18212890625),
 ('six', 6, 131072, 0.02, 'clean'): (-4.74853515625, -17.14111328125, 1.7578125)}

RATES = (.003, .020)
LABEL_CAP = 131072
REPLICATION_CAPS = (LABEL_CAP, 2 * LABEL_CAP)
TEACHER_COUNT = {"six": 5, "depth": 8, "replication": 16}
N_TEST = 8192


def read_counts():
    groups = {}
    for row in csv.DictReader(io.StringIO(COUNTS)):
        key = (row["study"], int(row["depth"]), int(row["labels"]), float(row["rate"]))
        values = groups.setdefault(key, [])
        values.append((int(row["teacher"]), int(row["final"]), int(row["clean"]),
                       int(row["noisy"]) if row["noisy"] else None))
    return groups



def trained_counts():
    """Read one count per teacher and arm at the figure's equal-label endpoints."""
    root = Path(__file__).resolve().parent
    groups = {}
    for key, peers in read_counts().items():
        study, depth, budget, rate = key
        arms = ("final", "clean", "noisy") if study == "replication" else ("final", "clean")
        for teacher, *_ in peers:
            if study == "six":
                path = root / "six_step_rates" / "results" / "fixed_curves.csv"
            elif study == "depth":
                path = root / "paired_depths" / "results" / f"metrics_{teacher}_{depth}.csv"
            else:
                path = root / "replication" / "results" / f"metrics_{teacher}.csv"
            counts = {}
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(row["seed"]) != teacher or float(row["rate"]) != rate:
                        continue
                    arm = row.get("arm") or row["supervision"]
                    arm = "clean" if arm == "value" else arm
                    if arm not in arms or int(row["n"]) * (1 + (arm != "final")) != budget:
                        continue
                    if arm in counts:
                        raise ValueError(f"Duplicate endpoint for {teacher}, {arm} in {path}")
                    if study == "six":
                        counts[arm] = round(float(row["clean_error"]) * N_TEST)
                    else:
                        assert int(row["labels"]) == budget and int(row["test"]) == N_TEST
                        assert int(row["depth"]) == depth
                        counts[arm] = int(row["errors"])
            if set(counts) != set(arms):
                raise ValueError(f"Missing arms at {key} for teacher {teacher}: {set(arms) - counts.keys()}")
            groups.setdefault(key, []).append((teacher, counts["final"], counts["clean"], counts.get("noisy")))
    return groups


def paired_depth(groups):
    """Exact paired percentile bootstrap for the exploratory 12-minus-9 step contrast."""
    for rate in RATES:
        errors = {}
        for depth in (9, 12):
            errors[depth] = {teacher: clean - final for teacher, final, clean, _ in
                             groups["depth", depth, LABEL_CAP, rate]}
        assert errors[9].keys() == errors[12].keys()
        changes = [errors[12][t] - errors[9][t] for t in errors[9]]
        n = len(changes)
        mass = {0: 1}
        for _ in changes:
            next_mass = Counter()
            for value, weight in mass.items():
                for change in changes:
                    next_mass[value + change] += weight
            mass = next_mass
        support = sorted(mass)
        cumulative = np.cumsum([mass[x] for x in support])
        # Percentile endpoints of the discrete bootstrap distribution.
        lo, hi = [support[np.searchsorted(cumulative, q * n**n)] for q in (.025, .975)]
        scale = 100 / (n * N_TEST)
        print(f"Rate {rate:.3f}, 12 minus 9 steps (pp): {sum(changes)*scale:.2f} "
              f"[{lo*scale:.2f}, {hi*scale:.2f}]")


def intervals(groups, archived):
    data = {}
    for key, counts in groups.items():
        study, depth, budget, rate = key
        counts.sort()
        n = TEACHER_COUNT[study]
        assert len(counts) == len({row[0] for row in counts}) == n
        if study != "replication":
            rng = random.Random(721001)
            rows = np.array([rng.choices(range(n), k=n) for _ in range(50000)])
            arms = ("clean",)
        else:
            rows = np.random.default_rng(7319011).integers(n, size=(50000, n))
            arms = ("clean", "noisy")
        final = np.array([row[1] for row in counts])
        for column, arm in enumerate(arms, 2):
            augmented = np.array([row[column] for row in counts])
            diff = 100.0 * (augmented - final) / N_TEST
            lo, hi = np.quantile(diff[rows].mean(axis=1), [.025, .975])
            result = (diff.mean(), lo, hi)
            k = (*key, arm)
            if archived:
                assert np.allclose(result, EXPECTED[k], rtol=0, atol=1e-10), k
            data[k] = result
    assert len(data) == len(EXPECTED) == 14
    return data


INK = HexColor("#20252B")
GRID = HexColor("#D7DCDF")
CLEAN = HexColor("#184F76")
NOISY = HexColor("#854616")
WHITE = Color(1, 1, 1)


def label(pdf, x, y, text, size=9.2, bold=False, align="left"):
    pdf.setFont("TimesNewRomanBold" if bold else "TimesNewRoman", size)
    pdf.setFillColor(INK)
    method = {"left": pdf.drawString, "right": pdf.drawRightString,
              "center": pdf.drawCentredString}[align]
    method(x, y, text)


def symbol(pdf, x, y, arm):
    color = CLEAN if arm == "clean" else NOISY
    pdf.setStrokeColor(color)
    if arm == "clean":
        pdf.setFillColor(color)
        pdf.circle(x, y, 2.35, stroke=1, fill=1)
    else:
        pdf.setFillColor(WHITE)
        path = pdf.beginPath()
        path.moveTo(x, y + 2.9)
        path.lineTo(x + 2.9, y)
        path.lineTo(x, y - 2.9)
        path.lineTo(x - 2.9, y)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)


def draw_panel(pdf, data, origin, study):
    depth_study = study == "depth"
    if depth_study:
        title = "(a) Equal-label depth comparison"
        subtitle = "131,072 training labels per arm"
        left_header = "Steps (n)"
        conditions = [(d, LABEL_CAP, r) for d in (6, 9, 12) for r in RATES]
        first_column = (24, 53)
    else:
        title = "(b) Independent replication, 16 teachers"
        subtitle = "Twelve steps at each label budget"
        left_header = "Labels"
        conditions = [(12, b, r) for b in REPLICATION_CAPS for r in RATES]
        first_column = (12, 42)
    label(pdf, origin, 162, title, 11, True)
    label(pdf, origin, 148, subtitle, 9.9)
    lcenter, rcenter = (origin + offset for offset in first_column)
    label(pdf, lcenter, 132, left_header, 9, True, "center")
    label(pdf, rcenter + 8, 132, "Rate", 9, True, "center")
    plot_left, plot_right = origin + 78, origin + 231
    x = lambda value: plot_left + (value + 40) * (plot_right - plot_left) / 45
    row_y = [117, 104, 88, 75, 59, 46] if depth_study else [114, 94, 72, 52]
    for tick in (-40, -30, -20, -10, 0):
        pdf.setLineWidth(.9 if tick == 0 else .35)
        pdf.setStrokeColor(INK if tick == 0 else GRID)
        pdf.line(x(tick), 40, x(tick), 124)
        label(pdf, x(tick), 33, str(tick).replace("-", "−"), 9, align="center")
    for (depth, budget, rate), y in zip(conditions, row_y):
        cohort = "six" if depth == 6 else study
        left_text = f"{depth} ({TEACHER_COUNT[cohort]})" if depth_study else f"{budget:,}"
        label(pdf, lcenter, y - 3, left_text, 9.4, align="center")
        label(pdf, rcenter + 8, y - 3, f"{rate:.3f}"[1:], 9.4, align="center")
        arms = (("clean", 0),) if depth_study else (("clean", 3.5), ("noisy", -3.5))
        for arm, offset in arms:
            mean, lo, hi = data[cohort, depth, budget, rate, arm]
            height = y + offset
            pdf.setStrokeColor(CLEAN if arm == "clean" else NOISY)
            pdf.setLineWidth(.85)
            pdf.line(x(lo), height, x(hi), height)
            for edge in (lo, hi):
                pdf.line(x(edge), height - 2, x(edge), height + 2)
            symbol(pdf, x(mean), height, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", action="store_true", help="read generated metrics under results/")
    args = parser.parse_args()
    fonts = Path(os.environ.get("TIMES_FONT_DIR", r"C:\Windows\Fonts"))
    if all((fonts / name).is_file() for name in ("times.ttf", "timesbd.ttf")):
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(fonts / "times.ttf")))
        pdfmetrics.registerFont(TTFont("TimesNewRomanBold", str(fonts / "timesbd.ttf")))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font("TimesNewRoman", "Times-Roman", "WinAnsiEncoding"))
        pdfmetrics.registerFont(pdfmetrics.Font("TimesNewRomanBold", "Times-Bold", "WinAnsiEncoding"))
    filename = "adam_label_depth_trained.pdf" if args.trained else "adam_label_depth.pdf"
    output = Path(__file__).with_name(filename)
    pdf = canvas.Canvas(str(output), pagesize=(488, 178), invariant=1,
                        initialFontName="TimesNewRoman", pageCompression=1)
    pdf.setTitle("Paired Adam comparisons at equal label cost")
    pdf.setAuthor("")
    groups = trained_counts() if args.trained else read_counts()
    data = intervals(groups, archived=not args.trained)
    paired_depth(groups)
    draw_panel(pdf, data, 8, "depth")
    draw_panel(pdf, data, 248, "replication")
    label(pdf, 244, 18.9, "Clean parity error difference (augmented minus final-only, percentage points)", 10.2, align="center")
    pdf.setLineWidth(.85)
    symbol(pdf, 132, 7, "clean")
    label(pdf, 139, 3.8, "Clean first sign", 9.4)
    symbol(pdf, 231, 7, "noisy")
    label(pdf, 238, 3.8, "First sign with 10% flips", 9.4)
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
