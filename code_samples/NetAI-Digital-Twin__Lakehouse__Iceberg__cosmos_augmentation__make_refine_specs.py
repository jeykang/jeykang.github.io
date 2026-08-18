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

# CONDITION-ONLY prompts: describe ONLY lighting/weather/road-surface/sky/atmosphere,
# NEVER vehicles/people/objects. Mentioning agents makes Cosmos hallucinate them on
# sparse scenes -> unlabeled objects -> invalid obstacle.offline labels (batch 2026-06-29).
# Depth control preserves existing geometry; the prompt only re-renders appearance.
PROMPTS = {
    "night": ("The video is captured from a forward-facing camera on a car driving at night. The "
              "scene is dark, lit only by ambient street lighting and the faint glow of the night "
              "sky. Streetlamps cast soft pools of warm light on the road surface, and areas beyond "
              "them fall into deep shadow. The asphalt is dark with faint reflections of the "
              "overhead lighting. The sky is black. The overall atmosphere is low-light and "
              "high-contrast, characteristic of nighttime, with no daylight."),
    "rain": ("The video is captured from a forward-facing camera on a car during steady rainfall "
             "in the daytime. The road surface is wet, glossy, and reflective, with a thin sheen of "
             "water and scattered shallow puddles. Falling rain and water droplets streak across "
             "the view. The sky is heavily overcast and gray. Colors are desaturated and the scene "
             "is low-contrast and muted, with everything glistening with moisture."),
    "fog": ("The video is captured from a forward-facing camera on a car driving through dense fog "
            "in the daytime. A thick, uniform gray-white haze blankets the scene, sharply reducing "
            "visibility so the road and surroundings dissolve into the mist a short distance ahead. "
            "The air is pale and featureless. The overall image is washed out, low-contrast, and "
            "muted."),
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
