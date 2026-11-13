"""Run HUGSIM's data/utils/estimate_depth.py with UniDepth's xformers paths switched off.

The PyPI xformers wheel HUGSIM's lock pulls (0.0.28.post1) is built for CUDA 12.1 and has
no usable kernels under the environment's torch 2.4.1+cu118. UniDepth checks module-level
XFORMERS_AVAILABLE flags at call time and otherwise uses PyTorch attention, so turning the
flags off after import is enough (the package itself must stay importable: other UniDepth
modules import its pure-Python parts).

    python estimate_depth_noxf.py --out <source_dir>     (cwd: HUGSIM repo/data)
"""
import os
import runpy
import sys

import unidepth.models.backbones.metadinov2.attention as attention
import unidepth.models.backbones.metadinov2.block as block

attention.XFORMERS_AVAILABLE = False
block.XFORMERS_AVAILABLE = False
sys.argv = ["estimate_depth.py"] + sys.argv[1:]
runpy.run_path(os.path.join(os.getcwd(), "utils", "estimate_depth.py"), run_name="__main__")
