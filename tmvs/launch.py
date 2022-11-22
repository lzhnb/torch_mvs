# Copyright (c) Zhihao Liang. All rights reserved.
import os
import argparse

import numpy as np
from tqdm import trange

from . import libmvs as _C

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert colmap camera")

    parser.add_argument(
        "--result_folder",
        "-rf",
        required=True,
        type=str,
        help="result_folder to store the dense results of COLMAP.",
    )
    parser.add_argument(
        "--geom_iterations",
        "-gi",
        type=int,
        default=1,
        help="geometric consistent iterations.",
    )
    parser.add_argument(
        "--planar_prior",
        "-pp",
        action="store_true",
        help="planar prior initilization",
    )
    parser.add_argument(
        "--geom_cons",
        "-gc",
        type=int,
        default=2,
        help="number of geometric consistent to filter the fusion point cloud, default to 2",
    )
    args = parser.parse_args()

    result_folder = args.result_folder
    problems = _C.generate_sample_list(os.path.join(result_folder, "pair.txt"))

    os.makedirs(os.path.join(result_folder, "ACMP"), exist_ok=True)

    num_images = len(problems)
    print(f"There are {num_images} problems needed to be processed!")

    pmmvs = _C.PMMVS()
    pmmvs.load_samples(result_folder, problems)
    print(f"Loaded all samples!")

    for i in trange(num_images, desc="initialization"):
        _C.process_problem(result_folder, problems[i], False, args.planar_prior, False, pmmvs)
    
    pmmvs.load_depths(result_folder, problems)
    print(f"Loaded all depths!")

    for geom_iter in range(args.geom_iterations):
        multi_geometry = geom_iter != 0
        for i in trange(num_images, desc="geometric consistent"):
            _C.process_problem(result_folder, problems[i], True, False, multi_geometry, pmmvs)
    
    depths, normals = _C.run_fusion(result_folder, problems, True, args.geom_cons)

    os.makedirs(os.path.join(result_folder, "depth_normal"), exist_ok=True)
    for i, depth, normal in zip(range(num_images), depths, normals):
        depth = depth[..., None]
        depth_normal = np.concatenate([depth, normal], axis=-1)
        save_file = os.path.join(result_folder, "depth_normal", f"{i:04}.npy")
        np.save(save_file, depth_normal)
