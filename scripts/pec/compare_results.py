import json
from collections import Counter

with open('logs/pec/dbg_fullrun_1/baseline_eval_results.json') as f:
    baseline = json.load(f)
with open('logs/pec/dbg_fullrun_1/expert0_iter3_md3300_results.json') as f:
    expert0 = json.load(f)
with open('logs/pec/dbg_fullrun_1/expert1_iter3_md3584_results.json') as f:
    expert1 = json.load(f)

bl_results = baseline['results']   # 400 designs
e0_results = expert0['results']    # 400 designs
e1_results = expert1['results']    # 400 designs

bl_scores  = [r['best_level_idx'] for r in bl_results]
e0_scores  = [r['best_level_idx'] for r in e0_results]
e1_scores  = [r['best_level_idx'] for r in e1_results]
pec_scores = [max(s0, s1) for s0, s1 in zip(e0_scores, e1_scores)]

N = len(bl_scores)
print("=== ALL 400 designs — Baseline vs max(Expert0, Expert1) ===")
print()

bl_mean  = sum(bl_scores)  / N
pec_mean = sum(pec_scores) / N
bl_med   = sorted(bl_scores)[N // 2]
pec_med  = sorted(pec_scores)[N // 2]

print("{:<28} {:>10} {:>12} {:>10}".format("Metric", "Baseline", "PEC (max)", "Delta"))
print('-' * 62)
print("{:<28} {:>10.2f} {:>12.2f} {:>+10.2f}".format("Mean level",   bl_mean,      pec_mean,      pec_mean - bl_mean))
print("{:<28} {:>10d} {:>12d} {:>+10d}".format("Median level", bl_med,       pec_med,       pec_med  - bl_med))
print("{:<28} {:>10d} {:>12d} {:>+10d}".format("Min level",    min(bl_scores), min(pec_scores), min(pec_scores)-min(bl_scores)))
print("{:<28} {:>10d} {:>12d} {:>+10d}".format("Max level",    max(bl_scores), max(pec_scores), max(pec_scores)-max(bl_scores)))

pec_better = sum(1 for b, p in zip(bl_scores, pec_scores) if p > b)
bl_better  = sum(1 for b, p in zip(bl_scores, pec_scores) if b > p)
tied       = sum(1 for b, p in zip(bl_scores, pec_scores) if b == p)
print("{:<28} {:>10} {:>12d}".format("PEC wins",      "-",        pec_better))
print("{:<28} {:>10d} {:>12}".format("Baseline wins", bl_better,  "-"))
print("{:<28} {:>10d}".format("Tied",                 tied))
print()

diffs = [p - b for b, p in zip(bl_scores, pec_scores)]
mean_diff = sum(diffs) / N
print("Mean improvement (PEC - Baseline): {:+.2f} levels".format(mean_diff))
print()

# Histogram
bucket_size = 3
buckets = Counter((d // bucket_size) * bucket_size for d in diffs)
print("Score difference histogram (PEC - Baseline), bucket=3 levels:")
for k in sorted(buckets):
    lo, hi = k, k + bucket_size - 1
    bar = '#' * buckets[k]
    if k > 0:
        tag = " <- PEC better"
    elif k < 0:
        tag = " <- Baseline better"
    else:
        tag = " <- tied/close"
    print("  [{:+3d},{:+3d}]: {:<35} ({:3d}){}".format(lo, hi, bar, buckets[k], tag))
print()

# Which expert was the winner for each design?
pec_winner = []
for s0, s1 in zip(e0_scores, e1_scores):
    if s0 >= s1:
        pec_winner.append(0)
    else:
        pec_winner.append(1)

e0_chosen = sum(1 for w in pec_winner if w == 0)
e1_chosen = sum(1 for w in pec_winner if w == 1)
print("PEC router: Expert 0 was best on {:d} designs, Expert 1 on {:d} designs.".format(e0_chosen, e1_chosen))
print()

# Top gains
print("Top 10 designs where PEC outperforms Baseline most:")
top_gains = sorted(zip(diffs, bl_results, pec_scores, pec_winner), key=lambda x: -x[0])[:10]
print("  {:>4}  {:>6}  {:>8}  {:>4}  {:>4}  {:>7}  {:>5}".format(
    "ID","GCR","spcf","BL","PEC","winner","Diff"))
for diff, bl_r, pec_s, winner in top_gains:
    print("  {:4d}  {:6.4f}  {:8.5f}  {:4d}  {:4d}  E{:<6d}  {:+5d}".format(
        bl_r['id'], bl_r['GCR'], bl_r['spcf'],
        bl_r['best_level_idx'], pec_s, winner, diff))
print()
print("Top 10 designs where Baseline outperforms PEC most:")
top_losses = sorted(zip(diffs, bl_results, pec_scores, pec_winner), key=lambda x: x[0])[:10]
for diff, bl_r, pec_s, winner in top_losses:
    print("  {:4d}  {:6.4f}  {:8.5f}  {:4d}  {:4d}  E{:<6d}  {:+5d}".format(
        bl_r['id'], bl_r['GCR'], bl_r['spcf'],
        bl_r['best_level_idx'], pec_s, winner, diff))
