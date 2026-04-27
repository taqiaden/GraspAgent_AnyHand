import math
import random
from packaging.version import Version
import matplotlib.pyplot as plt
import open3d
import torch
import trimesh
from utils.depth_map import depth_to_point_clouds, CameraInfo
from utils.image_utils import view_image
from utils.mesh_utils import construct_gripper_mesh_2
from utils.pc_utils import numpy_to_o3d
from utils.report_utils import distribution_summary
from utils.rl.masked_categorical import MaskedCategorical
from utils.pose_object import pose_7_to_transformation

parallel_jaw_model= 'new_gripper.ply'

object_prediction_threshold = 0.5
def plt_features(x,bins=100):
    for i in range(x.shape):
        x = x[:, i].detach().cpu().numpy()
        plt.hist(x, bins=100)
        plt.show()



def visualize_vox(npy):
    points_list = []
    for i in range(npy.shape[0]):
        for j in range(npy.shape[1]):
            for k in range(npy.shape[2]):
                # points_list.append((i, j, k))
                if npy[i, j, k] == 1: points_list.append((i, j, k))
    points_list = np.asarray(points_list)
    view_npy_open3d(points_list)


def view_o3d(pcd,view_coordinate=True,geometries_list=None):
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]) if view_coordinate else o3d.geometry.PointCloud()
    list=[] if geometries_list is None else geometries_list
    list.append(pcd)
    list.append(o)
    o3d.visualization.draw_geometries(list)
def view_o3d_objects(list_of_objects):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in list_of_objects:
        vis.add_geometry(obj)
        vis.run()
        vis.destroy_window()

def view_npy_open3d(pc,normals=None,color=None, view_coordinate=True,geometries_list=None):
    pcd = numpy_to_o3d(pc,normals=normals,color=color)
    view_o3d(pcd,view_coordinate,geometries_list)

def custom_normal_open3d_view(pc,normals=None,normal_mask=None,color=None, view_coordinate=True,geometries_list=None):
    pcd_with_normals= numpy_to_o3d(pc[normal_mask],normals=normals[normal_mask],color=color[normal_mask])
    pcd_without_normals= numpy_to_o3d(pc[~normal_mask],color=color[~normal_mask])

    o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]) if view_coordinate else o3d.geometry.PointCloud()
    list=[] if geometries_list is None else geometries_list
    list.append(pcd_with_normals)
    list.append(pcd_without_normals)
    list.append(o)
    o3d.visualization.draw_geometries(list)

def get_random_color():
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    return [r,g,b]

def view_npy_trimesh(npy_list,color_list=[],pick_random_colors=False):
    pc_=[]
    for i in range(len(npy_list)):
        if len(npy_list[i])<=1:continue
        # continue
        if len(color_list)>i:
            pc_.append(trimesh.PointCloud(npy_list[i], colors=color_list[i]))
        else:
            if pick_random_colors:
                pc_.append(trimesh.PointCloud(npy_list[i], colors=get_random_color()))
            else:
                pc_.append(trimesh.PointCloud(npy_list[i]))
    if pc_!=[]:
        scene_ = trimesh.Scene(pc_)
        scene_.show()

def o3d_line(start, end, colors_=None):
    points = [[start[0], start[1], start[2]],
              [end[0], end[1],
               end[2]]]
    lines = [[0, 1]]

    points = o3d.utility.Vector3dVector(points)
    lines = o3d.utility.Vector2iVector(lines)

    if colors_ is None: colors_=[0, 0.5, 0]
    colors = [colors_ for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def view_shift_pose(start,end,pc,target_normal,pc_colors=None):
    start=np.copy(start)
    end2=np.copy(start)
    end2+=target_normal*0.1
    vertical_line=o3d_line(start,end2,colors_=[0,0.5,0])

    pcd = numpy_to_o3d(pc,  color=pc_colors)

    o3d.visualization.draw_geometries([pcd, vertical_line])

def view_suction_zone(target_point,direction,pc,pc_colors):
    start=np.copy(target_point)
    end=np.copy(target_point)
    end=end+direction*0.1
    vertical_line = o3d_line(start, end, colors_=[0, 0.5, 0])
    pcd = numpy_to_o3d(pc, color=pc_colors)

    o3d.visualization.draw_geometries([pcd, vertical_line])





def visualize_detected_objects(objectness_pred, data_, object_prediction_threshold=object_prediction_threshold):
    objectness_pred_mask = objectness_pred > object_prediction_threshold
    ground_mask = ~objectness_pred_mask

    pc_objectness = trimesh.PointCloud(data_[objectness_pred_mask], colors=[0, 0, 255])
    pc_ground = trimesh.PointCloud(data_[ground_mask], colors=[0, 255, 0])
    scene_ = trimesh.Scene([pc_objectness, pc_ground])
    scene_.show()

def visualize_grasp_and_suction_points(suction_cls_pred_mask, grasp_cls_pred_mask, data_):
    grasp_suction_all = suction_cls_pred_mask & grasp_cls_pred_mask
    suction_only = suction_cls_pred_mask & ~grasp_cls_pred_mask
    grasp_only = grasp_cls_pred_mask & ~suction_cls_pred_mask

    postive_mask = suction_cls_pred_mask | grasp_cls_pred_mask
    negtive_mask = ~postive_mask

    scene_all = trimesh.Scene()

    if not True in grasp_suction_all:
        print('NO grasp_suction_all')
    else:
        pointcloud_blue = trimesh.PointCloud(data_[grasp_suction_all], colors=[0, 0, 255])
        scene_all.add_geometry(pointcloud_blue)

    if not True in suction_only:
        print('NO suction points')
    else:
        pointcloud_suction_only = trimesh.PointCloud(data_[suction_only], colors=[255, 165, 0])
        scene_all.add_geometry(pointcloud_suction_only)

    if not True in grasp_only:
        print('NO grasp points')
    else:
        pointcloud_grasp_only = trimesh.PointCloud(data_[grasp_only], colors=[255, 0, 255])
        scene_all.add_geometry(pointcloud_grasp_only)
    neg=data_[negtive_mask]
    if neg.shape[0]>0:
        pointcloud_blue_ = trimesh.PointCloud(neg, colors=[0, 255, 0])
        scene_all.add_geometry(pointcloud_blue_)
    scene_all.show()

def vis_depth_map(depth, view_as_point_cloud=True):
    if view_as_point_cloud:
        if isinstance(depth,torch.Tensor):
            depth=depth.numpy()
        camera = CameraInfo(480, 360, 1122.375, 1122.375, 296, 211, 1000)
        cloud,mask = depth_to_point_clouds(depth, camera)

        points = cloud.reshape(-1, 3)
        point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        point.transform(np.array([[0.010182, -0.999944, 0.003005, 0.39310000],
                           [-0.985716, -0.009532, 0.168148, -0.2809940],
                           [-0.168110, -0.004674, -0.985757, 1.3378300],
                           [0.0, 0.0, 0.0, 1.0]]))
        o3d.visualization.draw_geometries([point, axis_pcd])
    else:
        plt.imshow(depth, cmap='gray')
        plt.show()

def transform_coordinate(pc):
    matrix = np.array([[0.010182, -0.999944, 0.003005, 0.39310000],
                       [-0.985716, -0.009532, 0.168148, -0.2809940],
                       [-0.168110, -0.004674, -0.985757, 1.3378300],
                       [0.0, 0.0, 0.0, 1.0]])

    matrix_inv = np.linalg.inv(matrix)
    column = np.ones(len(pc))
    stacked = np.column_stack((pc, column))
    transformed = np.dot(matrix_inv, stacked.T).T[:, :3]
    transformed = np.ascontiguousarray(transformed)
    return transformed

import open3d as o3d
import numpy as np

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(end, origin, scale=1):
    assert(not np.all(end == origin))
    vec = end - origin

    size = np.sqrt(np.sum(vec**2))
    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    # Rz=Rz.cpu().numpy()
    # Ry=Ry.cpu().numpy()

    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))

    if  Version(open3d.__version__)>Version('0.15.2'):
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
    else:
        mesh.rotate(Ry, center=False)
        mesh.rotate(Rz, center=False)

    mesh.translate(origin)
    return(mesh)

def draw_arrow_implementation_example():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    arrow=get_arrow(origin=np.array([0, 0, 0]), end=np.array([1, 1, 1]), scale=1 / np.sqrt(3))
    vis.add_geometry(arrow)
    # vis.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame())
    vis.run()
    vis.destroy_window()

def score_to_color(score, RGB_variant=0):
    # print(np.isnan(score).any())
    # assert ~np.isnan(score).any()
    max_score,min_score,average,std=distribution_summary(score,data_name='Score')

    color=np.zeros((score.shape[0],3))
    for i in range(score.shape[0]):
        if score.shape[0]==1:color_intensity=0
        else:
            if max_score==min_score: min_score=min_score-0.00001
            color_intensity=math.floor((1-(score[i]-min_score)/(max_score-min_score))*255)
            color_intensity=min(color_intensity,255)
            color_intensity=max(color_intensity,0)

        color[i,RGB_variant]=255
        color[i,(RGB_variant+1)%3]=color_intensity
        color[i,(RGB_variant+2)%3]=color_intensity
    return color.astype(int)

def score_visualization(npy_points,npy_score):
    colors = score_to_color(npy_score, RGB_variant=0)
    p = trimesh.points.PointCloud(vertices=npy_points, colors=colors)
    p.show()

def view_score(data_,mask,score):
    masked_data = data_[mask]
    if masked_data.shape[0] == 0: return
    rest_of_data = data_[~mask]

    masked_score = score[mask]

    colors = score_to_color(masked_score, RGB_variant=0)

    p_data = trimesh.PointCloud(masked_data, colors=colors)
    if rest_of_data.shape[0] > 0: p_rest = trimesh.PointCloud(rest_of_data, colors=[255, 255, 255])

    scene = trimesh.Scene()
    scene.add_geometry(p_data)
    if rest_of_data.shape[0] > 0: scene.add_geometry(p_rest)

    scene.show()

def view_score2(data_,score):
    colors = score_to_color(score, RGB_variant=0)
    p_data = trimesh.PointCloud(data_, colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(p_data)
    scene.show()

if __name__ == '__main__':
    draw_arrow_implementation_example()
    x=np.random.random((10000,3))
    s=np.random.random((10000))
    score_visualization(x,s)

