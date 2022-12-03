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
        "--suffix",
        default="default",
        type=str,
        help="suffix directory to store the mvs results.",
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

    os.makedirs(os.path.join(result_folder, args.suffix), exist_ok=True)

    num_images = len(problems)
    print(f"There are {num_images} problems needed to be processed!")

    pmmvs = _C.PMMVS()
    pmmvs.load_samples(result_folder, problems)
    print(f"Loaded all samples!")

    for i in trange(num_images, desc="initialization"):
        depth_map, normal_map, cost_map = _C.process_problem(
            result_folder, problems[i], False, args.planar_prior, False, pmmvs
        )
        save_folder = os.path.join(result_folder, args.suffix, f"{i:04}")
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_folder, "depths.npy"), depth_map)
        np.save(os.path.join(save_folder, "normals.npy"), normal_map)
        np.save(os.path.join(save_folder, "costs.npy"), cost_map)

    for geom_iter in range(args.geom_iterations):
        multi_geometry = geom_iter != 0
        all_depths = []
        all_normals = []
        all_costs = []
        for i in trange(num_images, desc="loading for geometric consistent"):
            save_folder = os.path.join(result_folder, args.suffix, f"{i:04}")
            depth_suffix = "depths_geom.npy" if multi_geometry else "depths.npy"
            all_depths.append(np.load(os.path.join(os.path.join(save_folder, depth_suffix))))
            all_normals.append(np.load(os.path.join(os.path.join(save_folder, "normals.npy"))))
            all_costs.append(np.load(os.path.join(os.path.join(save_folder, "costs.npy"))))
        pmmvs.load_geometry(all_depths, all_normals, all_costs)

        # set geometry consistency parameters
        pmmvs.params.geom_consistency = True
        pmmvs.params.max_iterations = 2
        pmmvs.params.multi_geometry = multi_geometry
        all_depths = []
        all_normals = []
        for i in trange(num_images, desc="geometric consistent"):
            depth_map, normal_map, cost_map = _C.process_problem(
                result_folder, problems[i], True, False, multi_geometry, pmmvs
            )
            save_folder = os.path.join(result_folder, args.suffix, f"{i:04}")
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, "depths_geom.npy"), depth_map)
            np.save(os.path.join(save_folder, "normals.npy"), normal_map)
            np.save(os.path.join(save_folder, "costs.npy"), cost_map)
            all_depths.append(depth_map)
            all_normals.append(normal_map)

    depths, normals = _C.run_fusion(result_folder, problems, all_depths, all_normals, True, args.geom_cons)

    os.makedirs(os.path.join(result_folder, "depth_normal"), exist_ok=True)
    for i, depth, normal in zip(range(num_images), depths, normals):
        depth = depth[..., None]
        depth_normal = np.concatenate([depth, normal], axis=-1)
        save_file = os.path.join(result_folder, "depth_normal", f"{i:04}.npy")
        np.save(save_file, depth_normal)
