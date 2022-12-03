import argparse
import os
from typing import Optional

import numpy as np
from tqdm import tqdm


def vis_depth_norm(
    data_dir: str,
    depth_norm_dir: str,
    save_dir: str,
    H: float = 480,
    W: float = 640,
    filter: Optional[int] = None,
) -> None:
    # Reverse dirs
    cames_dir = os.path.join(data_dir, "pose")
    cames_list = os.listdir(cames_dir)
    cames_list.sort(key=lambda x: int(x.split(".")[0]))

    # Load dirs
    intrinsic = np.loadtxt(os.path.join(data_dir, "intrinsic.txt"))

    # Gen rays
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz = np.ones_like(xx)

    dirs = np.stack((xx, yy, zz), axis=-1)  # OpenCV convention
    dirs_ = dirs.reshape(-1, 3)  # [H * W, 3]

    # Get points
    points_all = []
    points_all_filter = []

    for cam_file in tqdm(cames_list):
        extrinsic = np.loadtxt(os.path.join(cames_dir, cam_file))
        depth_norm = np.load(os.path.join(depth_norm_dir, f"{int(cam_file[:-4]):04d}.npy"))
        origins = extrinsic[:3, 3]  # [3]
        origins = np.tile(origins, (H * W, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        depth = depth_norm[..., :1].reshape(-1, 1)
        normal = depth_norm[..., 1:].reshape(-1, 3)
        depth_mask = (depth > 0).squeeze()
        points = origins[depth_mask] + depth[depth_mask] * dirs[depth_mask]

        viewnorm_mask = ((-dirs[depth_mask]) * normal[depth_mask]).sum(-1) > np.cos(
            70 / 180 * np.pi
        )

        pointsn = np.concatenate((points, normal[depth_mask]), axis=1)
        points_all.append(pointsn)
        points_all_filter.append(pointsn[viewnorm_mask])

    points_all = np.concatenate(points_all)
    points_all_filter = np.concatenate(points_all_filter)

    num_points = points_all.shape[0]
    if filter is not None:
        if num_points > filter:
            sample_idx = np.random.choice(num_points, filter, False)
            points_all = points_all[sample_idx]

    num_points = points_all_filter.shape[0]
    if filter is not None:
        if num_points > filter:
            sample_idx = np.random.choice(num_points, filter, False)
            points_all_filter = points_all_filter[sample_idx]

    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(
        # os.path.join(save_dir, f"vis_depthnorm{suffix}.xyz"),
        os.path.join(save_dir, f"vis_depthnorm.xyz"),
        points_all,
        fmt="%.4f",
    )
    np.savetxt(
        # os.path.join(save_dir, f"vis_depthnorm{suffix}_filter.xyz"),
        os.path.join(save_dir, f"vis_depthnorm_filter.xyz"),
        points_all_filter,
        fmt="%.4f",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert colmap camera")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="dense_folder to store the dense results of COLMAP.",
    )
    parser.add_argument(
        "--depthnorm_dir",
        "-dn_dir",
        type=str,
        required=True,
        help="save_folder to store the results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="input_folder to store the predicted depth and normal.",
    )
    parser.add_argument("--height", "-H", type=int, default=480)
    parser.add_argument("--width", "-W", type=int, default=640)
    parser.add_argument(
        "--filter",
        "-f",
        type=int,
        default=None,
        help="filtered points number",
    )
    args = parser.parse_args()

    vis_depth_norm(
        args.data_dir, args.depthnorm_dir, args.save_dir, args.height, args.width, args.filter
    )
