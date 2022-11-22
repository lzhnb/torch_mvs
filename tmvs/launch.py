# Copyright (c) Zhihao Liang. All rights reserved.
import os
import argparse

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
    args = parser.parse_args()

    result_folder = args.result_folder
    problems = _C.generate_sample_list(os.path.join(result_folder, "pair.txt"))

    os.makedirs(os.path.join(result_folder, "ACMP"), exist_ok=True)

    num_images = len(problems)
    print(f"There are {num_images} problems needed to be processed!")

    for i in range(num_images):
        _C.process_problem(result_folder, problems[i], False, args.planar_prior, False)
    
    for geom_iter in range(args.geom_iterations):
        multi_geometry = geom_iter != 0
        for i in range(num_images):
            _C.process_problem(result_folder, problems[i], True, False, multi_geometry)
    
    _C.run_fusion(result_folder, problems, True)
