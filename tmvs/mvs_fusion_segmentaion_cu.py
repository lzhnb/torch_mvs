import argparse
import copy
import os
import time

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

import prim3d
from tmvs import _C

NTHEADS = 64
POISSON_MESH_THRESH = 0.03
CLEAN_PTS_THRESH = 0.03
MAX_FILTER_DEPTH = 10


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth_normal_dir",
        type=str,
        required=True,
        help="directory to store the MVS's depth and normal",
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
        "--ray_mask_dir",
        type=str,
        default=None,
        help="directory to store ray masks",
    )
    parser.add_argument(
        "--fliter_angle_thresh",
        type=float,
        default=70,
        help="the angle threshold to filter the points whose normals away from the view directions",
    )
    parser.add_argument(
        "--clean_mesh",
        action="store_true",
        help="clean the mesh (remove isolate components) or not",
    )
    parser.add_argument(
        "--clean_mesh_percentage",
        type=float,
        default=0.05,
        help="the percentage (isolate component ratio threshold) to clean mesh",
    )
    parser.add_argument(
        "--gen_mask",
        action="store_true",
        help="generate fusion mask (fusion textureless areas and superpixels) or not",
    )
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="textureless_mask",
        help="directory suffix to save generated fusion mask",
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
    o3d_mesh: o3d.geometry.TriangleMesh,
    diameter_percentage: float = 0.05,
    keep_mesh: bool = False,
    remove_unreferenced_vertices: bool = True,
) -> o3d.geometry.TriangleMesh:
    """remove the isolate components

    Args:
        o3d_mesh (o3d.geometry.TriangleMesh): input mesh
        diameter_percentage (float, optional): the ratio threshold of isolated components to be removed. Defaults to 0.05.
        keep_mesh (bool, optional): maintani the input mesh or not. Defaults to False.
        remove_unreferenced_vertices (bool, optional): remove the unreferenced vertices or not. Defaults to True.

    Returns:
        o3d.geometry.TriangleMesh: the cleaned mesh
    """

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
    """Visualize a single-channel image using matplotlib's viridis color map. yellow is high value, blue is low

    Args:
        gray (np.ndarray): input gray map, (H, W) or (H, W, 1) unscaled

    Returns:
        np.ndarray: (H, W, 3) float32 in [0, 1]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fusion parameters
    fusion_thresh = int(height * width / 32)
    non_textureless_percent = 0.75
    cos_sim_thresh = 0.8
    match_ratio_thresh = 0.75

    # make the directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "textureless_normal_clean"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth_normal_filter_clean"), exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(save_dir, "vis_clean"), exist_ok=True)
    if args.gen_mask:
        os.makedirs(os.path.join(save_dir, f"../../{args.mask_suffix}"), exist_ok=True)

    # get pose files
    pose_list = os.listdir(os.path.join(data_dir, "pose"))
    pose_list.sort(key=lambda x: int(x.split(".")[0]))
    num_imgs = len(pose_list)

    # get mvs depth&normal file
    depth_normal_list = os.listdir(depth_normal_dir)
    depth_normal_list.sort(key=lambda x: int(x.split(".")[0]))
    assert num_imgs == len(depth_normal_list)

    # get ray mask, the ray mask is used to filter out the background scene for outdoor scenes
    if ray_mask_dir is not None:
        ray_mask_list = os.listdir(ray_mask_dir)
        ray_mask_list = [rmf for rmf in ray_mask_list if rmf.endswith(".npy")]
        ray_mask_list.sort(key=lambda x: int(x.split(".")[0]))
        assert num_imgs == len(ray_mask_list)

        all_ray_masks = []
        for rmf in tqdm(ray_mask_list, desc="loading ray masks"):
            all_ray_masks.append(torch.from_numpy(np.load(os.path.join(ray_mask_dir, rmf))))
        all_ray_masks = torch.stack(all_ray_masks).to(device)

    # load all poses and depth&normals
    all_poses = []
    all_depth_normals = []
    for pose_file, depth_normal_file in tqdm(
        zip(pose_list, depth_normal_list), desc="loading poses and depth&normal"
    ):
        all_poses.append(torch.from_numpy(np.loadtxt(os.path.join(data_dir, "pose", pose_file))))
        all_depth_normals.append(
            torch.from_numpy(np.load(os.path.join(depth_normal_dir, depth_normal_file)))
        )
    all_poses = torch.stack(all_poses).to(device)
    all_depth_normals = torch.stack(all_depth_normals).to(device)

    # get intrinsic parameters
    intrinsic = torch.from_numpy(np.loadtxt(os.path.join(data_dir, "intrinsic.txt"))).float()
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # generate canonical directions
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
    dirs_ = dirs.reshape(-1, 3).to(device)  # [H * W, 3]

    """get mvs points"""
    points_all = []
    points_all_filter = []
    # the (angle) threshold between view directions and normals
    view_normal_thresh = np.cos(float(args.fliter_angle_thresh) / 180 * np.pi)
    for idx, extrinsic in tqdm(enumerate(all_poses), desc="gather mvs points"):
        # get extrinsic (pose) and generate the origins and directions
        origins = extrinsic[:3, 3]  # [3]
        origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        depth_normal = all_depth_normals[idx]
        depth = depth_normal[..., :1].reshape(-1, 1)
        normal = depth_normal[..., 1:].reshape(-1, 3)
        depth_mask = (depth > 0).squeeze()
        points = origins[depth_mask] + depth[depth_mask] * dirs[depth_mask]

        # filter out the points if the angles between view directions
        # and their normals are  larger then 70 degree
        view_normal_mask = ((-dirs[depth_mask]) * normal[depth_mask]).sum(-1) > view_normal_thresh
        if ray_mask_dir is not None:  # for the outdoor and dtu
            ray_mask = all_ray_masks[idx]
            assert ray_mask.shape == (height, width)
            depth_normal[torch.where(~ray_mask)] = 0.0
            bbox_diag = 1.1
            bbox_mask = torch.norm(points, p=2, dim=-1) < bbox_diag
            view_normal_mask = view_normal_mask * bbox_mask

        # get the valid mask (combine the valid depth mask and the view-normal-angle-filter mask)
        depth_normal_mask = torch.where(depth_mask == True, 1, 0)
        depth_normal_mask[torch.where(depth_normal_mask > 0.5)] = view_normal_mask.to(
            depth_normal_mask.dtype
        )
        depth_normal_mask = depth_normal_mask[..., None].expand(-1, 4).reshape(height, width, 4)
        depth_normal_filter = torch.where(
            depth_normal_mask > 0.5, depth_normal, torch.Tensor([0.0]).to(device)
        )
        all_depth_normals[idx] = depth_normal_filter
        # get the points with normals
        pointsn = torch.cat((points, normal[depth_mask]), dim=1)
        points_all.append(pointsn)
        points_all_filter.append(pointsn[view_normal_mask])

    # cat all pts
    points_all = torch.cat(points_all).cpu().numpy()
    points_all_filter = torch.cat(points_all_filter).cpu().numpy()
    if args.vis:
        # save mvs points
        np.savetxt(
            os.path.join(save_dir, "mvs_depthnorm.xyz"),
            points_all,
            fmt="%.4f",
        )
        np.savetxt(
            os.path.join(save_dir, "mvs_depthnorm_filter.xyz"),
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

        points_from_mesh_poisson = o3d.geometry.PointCloud(mesh_poisson.vertices)
        dist = pcd.compute_point_cloud_distance(points_from_mesh_poisson)
        dist = np.asarray(dist)
        filter_pts_clean_mask = torch.from_numpy(dist < POISSON_MESH_THRESH).to(device)
        print(
            f"Cleaned depth normal : {filter_pts_clean_mask.sum()} / {len(filter_pts_clean_mask)}"
        )

        all_depth_normals_mask = (all_depth_normals[..., 0] > 0).reshape(-1)
        all_depth_normals_mask[torch.where(all_depth_normals_mask)] = filter_pts_clean_mask
        all_depth_normals_mask = all_depth_normals_mask.reshape(-1, height, width)
        all_depth_normals[torch.where(~all_depth_normals_mask)] = 0.0
        if args.vis:
            points_all_filter_clean = points_all_filter[filter_pts_clean_mask.cpu().numpy()]
            # save mvs points
            np.savetxt(
                os.path.join(save_dir, "mvs_depth_normal_filter_clean.xyz"),
                points_all_filter_clean,
                fmt="%.4f",
            )

    """render and fusion mask"""
    vertices = np.asarray(mesh_poisson.vertices)
    faces = np.asarray(mesh_poisson.triangles)
    vertices_rc = torch.from_numpy(vertices).float().cuda()
    faces_rc = torch.from_numpy(faces).to(torch.int32).cuda()
    RT = prim3d.libPrim3D.create_raycaster(vertices_rc, faces_rc)
    torch.cuda.empty_cache()

    for depth_noraml, extrinsic, pose_file in tqdm(
        zip(all_depth_normals, all_poses, pose_list), desc="render textureless depth and normal"
    ):
        prefix = pose_file[:-4]
        np.save(
            os.path.join(save_dir, "depth_normal_filter_clean", f"{prefix}.npy"),
            depth_noraml.float().cpu().numpy(),
        )

        # get the camera extrinsic to get the view direction for ray casting
        origins = extrinsic[:3, 3]  # [3]
        origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        # ray casting to get the dense normals from reconstructed mesh, the dense
        # normals are used to generate pseudo normal map for textureless areas
        num_rays = origins.shape[0]
        filter_normal = torch.zeros_like(origins)
        filter_depth = torch.zeros([num_rays], dtype=torch.float32, device="cuda")
        primitive_ids = torch.zeros([num_rays], dtype=torch.int32, device="cuda") - 1
        RT.invoke(origins, dirs, filter_depth, filter_normal, primitive_ids)
        filter_normal = F.normalize(filter_normal, p=2, dim=-1).clamp(-1.0, 1.0)
        filter_depth = filter_depth.reshape(height, width)
        filter_normal = filter_normal.reshape(height, width, 3)
        filter_depth[filter_depth > MAX_FILTER_DEPTH] = 0
        filter_normal[filter_depth > MAX_FILTER_DEPTH] = 0
        filter_depth[torch.where(filter_depth < 0.05)] = 0
        np.save(
            os.path.join(save_dir, "textureless_normal_clean", f"{prefix}.npy"),
            filter_normal.cpu().numpy(),
        )

        filter_depth = filter_depth.cpu().numpy()
        filter_normal = filter_normal.cpu().numpy()
        textureless_area = np.where(filter_depth < 0.01, 1, 0)
        seg_ids = np.load(os.path.join(superpixel_dir, f"segid_npy/{prefix}.npy")).astype(np.int32)

        if args.gen_mask:
            fusion_mask = _C.fusion_textureless_mask(
                seg_ids,
                textureless_area,
                filter_normal,
                height,
                width,
                fusion_thresh,
                non_textureless_percent,
                cos_sim_thresh,
                match_ratio_thresh,
            )
            seg_ids = np.where((fusion_mask > 300), fusion_mask, seg_ids)
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
            vis_path = os.path.join(save_dir, "vis_clean", f"{prefix}.png")
            cv2.imwrite(vis_path, output_img)

    time_toc = time.time() - time_tic
    print("Time Used for Pre: ", time_toc)
