import argparse
import copy
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

from tmvs import _C

NTHEADS = 64
POISSON_MESH_THRESH = 0.03
CLEAN_PTS_THRESH = 0.03


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth_normal_dir",
        type=str,
        required=True,
        help="directory to store the MVS's depth and normal",
    )
    parser.add_argument(
        "--ray_mask_dir",
        type=str,
        default=None,
        help="directory to store ray masks",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="directory to store origin datas",
    )
    parser.add_argument(
        "--superpixel_dir",
        "-spd",
        type=str,
        required=True,
        help="directory to store the superpixel results",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        required=True,
        help="directory to store the output results",
    )
    parser.add_argument(
        "--fliter_angle_thresh",
        type=float,
        default=70,
        help="save visualization",
    )
    parser.add_argument(
        "--clean_mesh",
        action="store_true",
        help="save visualization",
    )
    parser.add_argument(
        "--clean_mesh_percentage",
        type=float,
        default=0.05,
        help="save visualization",
    )
    parser.add_argument(
        "--gen_mask",
        action="store_true",
        help="save visualization",
    )
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="textureless_mask",
        help="directory suffix to save visualization",
    )
    parser.add_argument("--height", type=int, default=480, help="the height of input images")
    parser.add_argument("--width", type=int, default=640, help="the width of input images")
    parser.add_argument(
        "--vis",
        action="store_true",
        help="save visualization",
    )
    args = parser.parse_args()
    return args


def remove_isolate_component_by_diameter(
    o3d_mesh,
    diameter_percentage: float = 0.05,
    keep_mesh: bool = False,
    remove_unreferenced_vertices: bool = True,
):
    import copy

    assert diameter_percentage >= 0.0
    assert diameter_percentage <= 1.0
    max_bb = o3d_mesh.get_max_bound()
    min_bb = o3d_mesh.get_min_bound()
    size_bb = np.abs(max_bb - min_bb)
    filter_diag = diameter_percentage * np.linalg.norm(size_bb)

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, _ = o3d_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters) + 1
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = cluster_n_triangles.argmax() + 1
    for idx in range(1, len(cluster_n_triangles) + 1):  # set label 0 to keep
        if idx == largest_cluster_idx:  # must keep the largest
            triangle_clusters[triangle_clusters == idx] = 0
        else:
            cluster_triangle = triangle_clusters == idx
            cluster_index = np.unique(faces[cluster_triangle])
            cluster_vertices = vertices[cluster_index]
            cluster_bbox = np.abs(
                np.amax(cluster_vertices, axis=0) - np.amin(cluster_vertices, axis=0)
            )
            cluster_bbox_diag = np.linalg.norm(cluster_bbox, ord=2)
            if cluster_bbox_diag >= filter_diag:
                triangle_clusters[triangle_clusters == idx] = 0
    mesh_temp = copy.deepcopy(o3d_mesh) if keep_mesh else o3d_mesh
    mesh_temp.remove_triangles_by_mask(triangle_clusters > 0.5)

    if remove_unreferenced_vertices:
        mesh_temp.remove_unreferenced_vertices()
    return mesh_temp


def viridis_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)


if __name__ == "__main__":
    time_tic = time.time()

    # read arguments
    args = get_args()
    depth_normal_dir = args.depth_normal_dir
    data_dir = args.data_dir
    ray_mask_dir = args.ray_mask_dir
    superpixel_dir = args.superpixel_dir
    save_dir = args.save_dir
    height = args.height
    width = args.width

    # make the directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth_normal_filter"), exist_ok=True)
    if args.gen_mask:
        os.makedirs(os.path.join(save_dir, f"../../{args.mask_suffix}"), exist_ok=True)
    if args.clean_mesh:
        os.makedirs(os.path.join(save_dir, "textureless_normal_clean"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "depth_normal_filter_clean"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "vis_clean"), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_dir, "textureless_normal"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "vis"), exist_ok=True)

    # get pose files
    pose_list = os.listdir(os.path.join(data_dir, "pose"))
    pose_list.sort(key=lambda x: int(x.split(".")[0]))
    num_imgs = len(pose_list)

    # get mvs depth&normal file
    depth_normal_list = os.listdir(depth_normal_dir)
    depth_normal_list.sort(key=lambda x: int(x.split(".")[0]))
    assert num_imgs == len(depth_normal_list)

    # get intrinsic parameters
    intrinsic = np.loadtxt(os.path.join(data_dir, "intrinsic.txt"))
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # generate canonical directions
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz = np.ones_like(xx)
    dirs = np.stack((xx, yy, zz), axis=-1)  # OpenCV convention
    dirs_ = dirs.reshape(-1, 3)  # [H * W, 3]

    """get mvs points"""
    points_all = []
    points_all_filter = []
    # the (angle) threshold between view directions and normals
    view_normal_thresh = np.cos(float(args.fliter_angle_thresh) / 180 * np.pi)
    for pose_file in tqdm(pose_list, desc="Gather mvs points"):
        prefix = pose_file[:-4]

        # get extrinsic (pose) and generate the origins and directions
        extrinsic = np.loadtxt(os.path.join(data_dir, "pose", pose_file))
        origins = extrinsic[:3, 3]  # [3]
        origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        # read depth & normal and convert into point cloud
        depth_normal = np.load(os.path.join(depth_normal_dir, f"{prefix}.npy"))
        depth = depth_normal[..., :1].reshape(-1, 1)
        normal = depth_normal[..., 1:].reshape(-1, 3)
        depth_mask = (depth > 0).squeeze()
        points = origins[depth_mask] + depth[depth_mask] * dirs[depth_mask]

        # filter out the points if the angles between view directions
        # and their normals are  larger then 70 degree
        view_normal_mask = ((-dirs[depth_mask]) * normal[depth_mask]).sum(-1) > view_normal_thresh

        # get the valid mask (combine the valid depth mask and the view-normal-angle-filter mask)
        depth_normal_mask = np.where(depth_mask == True, 1, 0)
        depth_normal_mask[np.where(depth_normal_mask > 0.5)] = view_normal_mask.astype(np.int32)
        depth_normal_mask = (
            depth_normal_mask[..., None].repeat(4, axis=-1).reshape(height, width, 4)
        )
        depth_normal_filter = np.where(depth_normal_mask > 0.5, depth_normal, 0)
        # save the filtered depth normal
        np.save(
            os.path.join(save_dir, "depth_normal_filter", f"{prefix}.npy"),
            depth_normal_filter.astype(np.float32),
        )
        # get the points with normals
        pointsn = np.concatenate((points, normal[depth_mask]), axis=1)
        points_all.append(pointsn)
        points_all_filter.append(pointsn[view_normal_mask])

    # cat all pts
    points_all = np.concatenate(points_all)
    points_all_filter = np.concatenate(points_all_filter)

    # save mvs points
    np.savetxt(
        os.path.join(save_dir, "mvs_depth_normal.xyz"),
        points_all,
        fmt="%.4f",
    )
    np.savetxt(
        os.path.join(save_dir, "mvs_depth_normal_filter.xyz"),
        points_all_filter,
        fmt="%.4f",
    )

    """Points to Mesh"""
    points_normals = points_all_filter.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_normals[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points_normals[:, 3:])
    del points_all
    del points_all_filter

    # Poisson reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        (
            mesh_poisson,
            densities,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "poisson_mesh.ply"), mesh_poisson)

    """filter the reconstructed mesh (correspondent to the textureless areas)"""
    points_from_mesh_poisson = o3d.geometry.PointCloud(mesh_poisson.vertices)
    dist = points_from_mesh_poisson.compute_point_cloud_distance(pcd)
    dist = np.asarray(dist)
    ind = np.where(dist > POISSON_MESH_THRESH)[0]
    mesh_poisson.remove_vertices_by_index(ind)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "poisson_mesh_filter.ply"), mesh_poisson)

    """clean the mesh again ot remove isolate faces to, use the cleaned mesh to raycast scene"""
    if args.clean_mesh:
        mesh_poisson = remove_isolate_component_by_diameter(
            mesh_poisson, diameter_percentage=float(args.clean_mesh_percentage)
        )
        o3d.io.write_triangle_mesh(os.path.join(save_dir, "filter_clean.ply"), mesh_poisson)

    filter_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_poisson)
    filter_scene = o3d.t.geometry.RaycastingScene()
    _ = filter_scene.add_triangles(filter_mesh)

    """clean the depth and normal with the filter_cleane mesh"""
    if args.clean_mesh:
        points_all_filter_clean = []
        for pose_file in tqdm(pose_list, desc="Clean depth and normal"):
            prefix = pose_file[:-4]

            # get extrinsic (pose) and generate the origins and directions
            extrinsic = np.loadtxt(os.path.join(data_dir, "pose", pose_file))
            origins = extrinsic[:3, 3]  # [3]
            origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
            rot_mat = extrinsic[:3, :3]
            dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

            # read depth & normal and convert into point cloud
            depth_normal = np.load(os.path.join(save_dir, "depth_normal_filter", f"{prefix}.npy"))
            depth = depth_normal[..., :1].reshape(-1, 1)
            normal = depth_normal[..., 1:].reshape(-1, 3)
            depth_mask = (depth > 0).squeeze()
            points = origins[depth_mask] + depth[depth_mask] * dirs[depth_mask]

            # convert points into open3d tensor and compute distance
            query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
            pts_mesh_dist = filter_scene.compute_distance(query_point, nthreads=NTHEADS)
            del query_point
            pts_mesh_dist = np.asarray(pts_mesh_dist)
            pts_mesh_mask = pts_mesh_dist < CLEAN_PTS_THRESH
            del pts_mesh_dist
            # get the clean mask
            depth_normal_mask = np.where(depth_mask == True, 1, 0)
            depth_normal_mask[np.where(depth_normal_mask > 0.5)] = pts_mesh_mask.astype(np.int32)
            depth_normal_mask = (
                depth_normal_mask[..., None].repeat(4, axis=-1).reshape(height, width, 4)
            )
            depth_normal_filter = np.where(depth_normal_mask > 0.5, depth_normal, 0)
            np.save(
                os.path.join(save_dir, "depth_normal_filter_clean", f"{prefix}.npy"),
                depth_normal_filter.astype(np.float32),
            )

            pointsn = np.concatenate((points, normal[depth_mask]), axis=1)
            points_all_filter_clean.append(pointsn[pts_mesh_mask])

        # cat all pts
        points_all_filter_clean = np.concatenate(points_all_filter_clean)
        # save mvs points
        np.savetxt(
            os.path.join(save_dir, "mvs_depth_normal_filter_clean.xyz"),
            points_all_filter_clean,
            fmt="%.4f",
        )

    """render and fusion mask"""
    thresh = int(height * width / 32)
    non_textureless_percent = 0.75
    cos_sim_thresh = 0.8
    match_ratio_thresh = 0.75
    for pose_file in tqdm(pose_list, desc="fusion mask"):
        prefix = pose_file[:-4]

        # get the camera extrinsic to get the view direction for ray casting
        extrinsic = np.loadtxt(os.path.join(data_dir, "pose", pose_file))
        origins = extrinsic[:3, 3]  # [3]
        origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        # ray casting to get the dense normals from reconstructed mesh, the dense
        # normals are used to generate pseudo normal map for textureless areas
        rays = np.concatenate([origins, dirs], axis=1)  # [H * W, 6]
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        filter_ans = filter_scene.cast_rays(rays, nthreads=NTHEADS)
        filter_depth = filter_ans["t_hit"].numpy().reshape(height, width)
        filter_normal = filter_ans["primitive_normals"].numpy().reshape(height, width, 3)
        filter_depth[np.isinf(filter_depth)] = 0
        filter_normal[np.isinf(filter_depth)] = 0
        filter_depth[np.where(filter_depth < 0.05)] = 0

        if args.clean_mesh:
            np.save(os.path.join(save_dir, "textureless_normal_clean", f"{prefix}.npy"), filter_normal)
        else:
            np.save(os.path.join(save_dir, "textureless_normal", f"{prefix}.npy"), filter_normal)

        # load superpixel ids
        seg_ids = np.load(os.path.join(superpixel_dir, os.path.join("segid_npy", f"{prefix}.npy")))
        seg_ids_: np.ndarray = copy.deepcopy(seg_ids).astype(np.int32)
        # this mask is used to generate fusion mask
        textureless_area = np.where(filter_depth < 0.01, 1, 0)

        if args.gen_mask:
            # fusion the textureless area and the superpixels to get the mask
            fusion_mask = _C.fusion_textureless_mask(
                seg_ids_,
                textureless_area,
                filter_normal,
                height,
                width,
                thresh,
                non_textureless_percent,
                cos_sim_thresh,
                match_ratio_thresh,
            )
            seg_ids_ = np.where((fusion_mask > 300), fusion_mask, seg_ids_)
            fusion_mask[fusion_mask == 300] = 0  # for objects segs, remove them
            np.save(
                os.path.join(save_dir, f"../../{args.mask_suffix}", f"{prefix}.npy"),
                fusion_mask,
            )
        else:
            fusion_mask = np.zeros_like(textureless_area)

        if args.vis:
            # read image
            img = cv2.imread(os.path.join(data_dir, "images", f"{prefix}.png"))
            # vis the textureless mask image
            textureless_img = img.copy()
            textureless_img = np.where(textureless_area[..., None].repeat(3, axis=-1) > 0.5, img, 0)
            # vis the superpixel and fusion textureless mask image
            color_bar = np.random.randint(0, 255, [max(seg_ids.max(), fusion_mask.max()) + 1, 3])
            color_bar[0] = 0
            seg_ids_rgb = color_bar[seg_ids]
            fusion_mask_rgb = color_bar[fusion_mask]

            fusion_rgb = copy.deepcopy(fusion_mask_rgb)
            fusion_rgb[fusion_mask > 300] = 255
            fusion_rgb = np.where(fusion_mask[..., None].repeat(3, axis=-1) > 0, img, 0)

            # comprehensive visualization
            normal_rgb = ((filter_normal + 1) * 127.5).astype(np.uint8)
            output_img = np.concatenate(
                [
                    np.concatenate([img, textureless_img, fusion_mask_rgb], axis=1),
                    np.concatenate([seg_ids_rgb, normal_rgb, fusion_rgb], axis=1),
                ],
                axis=0,
            )
            if args.clean_mesh:
                vis_path = os.path.join(save_dir, "vis_clean", f"{prefix}.png")
            else:
                vis_path = os.path.join(save_dir, "vis", f"{prefix}.png")
            cv2.imwrite(vis_path, output_img)

    time_toc = time.time() - time_tic
    print("Time Used for Pre: ", time_toc)
