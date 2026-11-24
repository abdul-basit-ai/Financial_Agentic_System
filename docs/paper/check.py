import itertools
import re
from pathlib import Path

s = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
figure = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", s, re.DOTALL)
if figure is None:
    raise SystemExit("No TikZ architecture figure was found.")
fig1 = figure.group(0)

nodes = re.findall(
    r"\\node\[[^\]]*\]\s*\((\w+)\)\s*at\s*\(([\d.\-]+),\s*([\d.\-]+)\)", fig1
)
print("explicit-position nodes:", nodes)

approx = {}
for name, x, y in nodes:
    x, y = float(x), float(y)
    approx[name] = (x, y)


def w(name):
    label_len = 0
    m = re.search(
        rf"\\node\[[^\]]*\]\s*\({re.escape(name)}\)[^{{]*\{{([^}}]*)\}}", fig1
    )
    if m:
        label_len = len(m.group(1))
    return max(2.0, label_len * 0.10)


overlaps = []
for (a, ax, ay), (b, bx, by) in itertools.combinations(
    [(n, v[0], v[1]) for n, v in approx.items()], 2
):
    if abs(ay - by) < 0.75:
        gap = abs(ax - bx) - (w(a) + w(b)) / 2
        status = "OVERLAP!" if gap < -0.1 else ("tight" if gap < 0.25 else "ok")
        overlaps.append((a, b, round(gap, 2), status))
for o in overlaps:
    print(o)
print("rows distinct check done")
