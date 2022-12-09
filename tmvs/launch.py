# Copyright (c) Zhihao Liang. All rights reserved.
import argparse
import os
import shutil

import numpy as np
from tqdm import trange

from tmvs import _C


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert colmap camera")

    parser.add_argument(
        "--result_dir",
        "-rd",
        type=str,
        required=True,
        help="directory to store the dense results of COLMAP.",
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
        "--input_depth_normal_dir",
        "-idnd",
        type=str,
        default=None,
        help="directory to store the input depth and normal for initialization",
    )
    parser.add_argument(
        "--dn_input",
        "-di",
        action="store_true",
        help="if use depth and normal input for initialization, please make sure there is */depth.npy and */normal.npy under input_depth_normal_dir",
    )
    parser.add_argument(
        "--geom_cons",
        "-gc",
        type=int,
        default=2,
        help="number of geometric consistent to filter the fusion point cloud, default to 2",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    result_dir = args.result_dir
    problems = _C.generate_sample_list(os.path.join(result_dir, "pair.txt"))

    os.makedirs(os.path.join(result_dir, args.suffix), exist_ok=True)

    num_images = len(problems)
    print(f"There are {num_images} problems needed to be processed!")

    pmmvs = _C.PMMVS()
    pmmvs.load_samples(result_dir, problems)
    print(f"Loaded all samples!")

    # initialization
    for ref_id in trange(num_images, desc="Initialization"):
        save_dir = os.path.join(result_dir, args.suffix, f"{ref_id:04}")
        os.makedirs(save_dir, exist_ok=True)
        if args.dn_input:  # given initial depths and normals
            depth_normal_dir = os.path.join(args.input_depth_normal_dir, f"{ref_id:04}")
            shutil.copy2(
                os.path.join(depth_normal_dir, "depth.npy"), os.path.join(save_dir, "depths.npy")
            )
            shutil.copy2(
                os.path.join(depth_normal_dir, "normal.npy"),
                os.path.join(save_dir, "normals.npy"),
            )
            shutil.copy2(
                os.path.join(depth_normal_dir, "cost.npy"), os.path.join(save_dir, "costs.npy")
            )
        else:  # initialization from uniform distribution or triangular planar prior
            depth_map, normal_map, cost_map = _C.process_problem(
                result_dir, problems[ref_id], False, args.planar_prior, False, pmmvs
            )
            np.save(os.path.join(save_dir, "depths.npy"), depth_map)
            np.save(os.path.join(save_dir, "normals.npy"), normal_map)
            np.save(os.path.join(save_dir, "costs.npy"), cost_map)

    # geometric consistency
    for geom_iter in range(args.geom_iterations):
        multi_geometry = geom_iter != 0
        all_depths = []
        all_normals = []
        all_costs = []
        for ref_id in trange(num_images, desc="Loading for geometric consistent"):
            save_dir = os.path.join(result_dir, args.suffix, f"{ref_id:04}")
            depth_suffix = "depths_geom.npy" if multi_geometry else "depths.npy"
            all_depths.append(np.load(os.path.join(os.path.join(save_dir, depth_suffix))))
            all_normals.append(np.load(os.path.join(os.path.join(save_dir, "normals.npy"))))
            all_costs.append(np.load(os.path.join(os.path.join(save_dir, "costs.npy"))))
        pmmvs.load_geometry(all_depths, all_normals, all_costs)

        # set geometry consistency parameters
        pmmvs.params.geom_consistency = True
        pmmvs.params.max_iterations = 2
        pmmvs.params.multi_geometry = multi_geometry
        all_depths = []
        all_normals = []
        for ref_id in trange(num_images, desc="Geometric consistent"):
            depth_map, normal_map, cost_map = _C.process_problem(
                result_dir, problems[ref_id], True, False, multi_geometry, pmmvs
            )
            save_dir = os.path.join(result_dir, args.suffix, f"{ref_id:04}")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "depths_geom.npy"), depth_map)
            np.save(os.path.join(save_dir, "normals.npy"), normal_map)
            np.save(os.path.join(save_dir, "costs.npy"), cost_map)
            all_depths.append(depth_map)
            all_normals.append(normal_map)

    # run fusion
    depths, normals = _C.run_fusion(
        result_dir, problems, all_depths, all_normals, True, args.geom_cons
    )

    # save the inference results (depths and normals)
    os.makedirs(os.path.join(result_dir, args.suffix, "depth_normal"), exist_ok=True)
    for i, depth, normal in zip(range(num_images), depths, normals):
        depth = depth[..., None]
        depth_normal = np.concatenate([depth, normal], axis=-1)
        save_file = os.path.join(result_dir, args.suffix, "depth_normal", f"{i:04}.npy")
        np.save(save_file, depth_normal)
