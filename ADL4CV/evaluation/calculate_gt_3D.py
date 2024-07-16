import torch
import json
import mcubes
import trimesh
from ADL4CV.utils import *
import random

armadillo_combined_idx = [
    54,
    61,
    201,
    226,
    260,
    279,
    341,
    344,
    370,
    403,
    457,
    489,
    522,
    539,
    558,
    597,
    55,
    138,
    473,
    574,
]

armadillo_pairs = [
    [x, y] for x in armadillo_combined_idx for y in armadillo_combined_idx
]
random.seed(42)
random.shuffle(armadillo_pairs)

train_pairs = armadillo_pairs[:370]
val_pairs = armadillo_pairs[370:]

for pair in val_pairs:
    idx_1, idx_2 = pair
    generate_merged_gt_3D(
        idx_1,
        idx_2,
        np.array([0, -0.5, 0, 0, 0.5, 0]),
        f"./ADL4CV/evaluation/pairs_gt/merged_gt_{idx_1}_{idx_2}.obj",
    )
