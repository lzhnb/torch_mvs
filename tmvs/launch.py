# Copyright (c) Zhihao Liang. All rights reserved.
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
        default=2,
        help="geometric consistent iterations.",
    )
    parser.add_argument(
        "--planar_prior",
        "-pp",
        action="store_true",
        help="planar prior initilization",
    )
    args = parser.parse_args()

    _C.launch(args.result_folder, args.geom_iterations, args.planar_prior)
