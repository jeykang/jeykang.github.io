"""Unit smoke test for compute_scene_score's perception renormalization.

No Spark, no destructive writes. Just imports the function and checks
the active-weight renormalization behaves correctly.
"""
import sys
sys.path.insert(0, "/home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg")

from nvidia_ingestion.edge_case_scorer import compute_scene_score, _SCENE_WEIGHTS

print("Weights:", _SCENE_WEIGHTS)
print()

# Baseline: metadata-only (no ego, no obstacle, no perception)
sensor_flags = {"radar_corner_front_left_srr_0": True}
baseline_score, baseline_sub = compute_scene_score(
    hour=2, season="winter", country="Finland",
    sensor_flags=sensor_flags,
)
print(f"[1] metadata-only (no perception, no obstacle):")
print(f"    score = {baseline_score:.4f}")
print(f"    sub   = {baseline_sub}")

# Now with a high perception score
with_perc_score, with_perc_sub = compute_scene_score(
    hour=2, season="winter", country="Finland",
    sensor_flags=sensor_flags,
    perception_score=0.85,
)
print(f"\n[2] with perception=0.85:")
print(f"    score = {with_perc_score:.4f}")
print(f"    sub   = {with_perc_sub}")

# With a low perception score
with_low_perc_score, with_low_perc_sub = compute_scene_score(
    hour=2, season="winter", country="Finland",
    sensor_flags=sensor_flags,
    perception_score=0.1,
)
print(f"\n[3] with perception=0.10:")
print(f"    score = {with_low_perc_score:.4f}")
print(f"    sub   = {with_low_perc_sub}")

# With ego + perception (5 active dimensions)
full_score, full_sub = compute_scene_score(
    hour=2, season="winter", country="Finland",
    sensor_flags=sensor_flags,
    ego_score=0.7,
    perception_score=0.9,
)
print(f"\n[4] ego=0.70 + perception=0.90:")
print(f"    score = {full_score:.4f}")
print(f"    sub   = {full_sub}")

# Sanity: with all 6 dimensions present (obstacle + perception)
all_score, all_sub = compute_scene_score(
    hour=2, season="winter", country="Finland",
    sensor_flags=sensor_flags,
    ego_score=0.7,
    obstacle_count=15,
    perception_score=0.9,
)
print(f"\n[5] all 6 dims (obstacle=15, ego=0.7, perception=0.9):")
print(f"    score = {all_score:.4f}")
print(f"    sub   = {all_sub}")

# Verify renormalization: the score must be a weighted avg of non-None subs
def manual_score(sub, weights):
    active = [k for k, v in sub.items() if v is not None]
    total_w = sum(weights[k] for k in active)
    return sum(sub[k] * weights[k] / total_w for k in active)

for i, (s, sub) in enumerate([(baseline_score, baseline_sub),
                               (with_perc_score, with_perc_sub),
                               (with_low_perc_score, with_low_perc_sub),
                               (full_score, full_sub),
                               (all_score, all_sub)], 1):
    expected = manual_score(sub, _SCENE_WEIGHTS)
    ok = abs(s - expected) < 1e-9
    print(f"case {i}: got={s:.6f} expected={expected:.6f}  {'OK' if ok else 'MISMATCH'}")
