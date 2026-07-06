"""Generate a control x condition matrix of controlnet specs for one clip, to compare
realism / geometry-preservation / difficulty before committing to a batch recipe.

Axes:
  - control: depth (lighting-invariant, strong geometry) vs edge (appearance edges) vs
    multi (depth+edge+vis blended) — structure-preservation vs realism trade-off.
  - condition: night / rain / fog (prompt-driven appearance).
Writes specs to cosmos_augmentation/refine_specs/<name>.json (input = 121-frame clip).
"""
import json, os

CLIP = "/scratch/autodr_test/aug_test/inputs/02fd3a17_121.mp4"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refine_specs")
os.makedirs(OUT, exist_ok=True)

PROMPTS = {
    "night": ("The video is captured from a camera mounted on a car, facing forward, driving at "
              "night. The scene is dark, illuminated only by streetlights, traffic signals, and "
              "the headlights of vehicles. The asphalt reflects the warm glow of streetlights and "
              "the red and white lights of other cars. Oncoming vehicles cast bright headlight "
              "beams, while parked cars appear as dim silhouettes. Buildings and trees are dark "
              "shapes against a black night sky. Low-light, high-contrast, deep shadows and bright "
              "artificial light sources, characteristic of nighttime driving."),
    "rain": ("The video is captured from a camera mounted on a car driving in heavy rain during "
             "the day. The road is wet and covered with puddles reflecting the gray, overcast sky. "
             "Rain streaks across the windshield, partially blurring the view. Water spray rises "
             "from the tires of vehicles. Visibility is reduced; distant objects appear hazy "
             "through the falling rain. The asphalt glistens with moisture, and the overall scene "
             "is desaturated, gray, and low-contrast, typical of driving in a rainstorm."),
    "fog": ("The video is captured from a camera mounted on a car driving through dense fog. "
            "Visibility is severely reduced; the road ahead fades into a thick gray-white haze "
            "within a short distance. Distant vehicles, buildings, and trees are barely visible as "
            "faint silhouettes emerging from the fog. Headlights of oncoming cars appear as "
            "diffuse glowing halos. The scene is washed out, low-contrast, and muted."),
}

# control configs (name -> control dict)
CONTROLS = {
    "depth": {"depth": {"control_weight": 1.0}},
    "edge":  {"edge": {"control_weight": 1.0}},
    "multi": {"depth": {"control_weight": 0.5}, "edge": {"control_weight": 0.3},
              "vis": {"control_weight": 0.3}},
}

# matrix: night gets all 3 controls (control comparison); rain/fog use depth (validated)
COMBOS = [("night", "depth"), ("night", "edge"), ("night", "multi"),
          ("rain", "depth"), ("fog", "depth")]

for cond, ctrl in COMBOS:
    spec = {"prompt": PROMPTS[cond], "input_video_path": CLIP}
    spec.update(CONTROLS[ctrl])
    name = f"{cond}_{ctrl}"
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(spec, f, indent=2)
    print(f"wrote {name}.json  (control={ctrl})")
print(f"{len(COMBOS)} specs -> {OUT}")
