import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils, models
from scipy import ndimage
import copy
import time
import random
import math
import skimage.transform as trans
# from utils_warp import convert_image_np, normalize_transforms, rotatepoints, show_image



def get_pointcloud(color_img, depth_img, camera_intrinsics):
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Fake intrinsics for now

    # Project depth back into 3D point cloud in Camera coordinates (I think he means relative to camera frame)
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w-1, im_w), np.linspace(0, im_h-1, im_h))

    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2], depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2], depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()

    cam_pts_x.shape = (im_h*im_w, 1)
    cam_pts_y.shape = (im_h*im_w, 1)
    cam_pts_z.shape = (im_h*im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0].reshape((im_h*im_w, 1))
    rgb_pts_g = color_img[:, :, 1].reshape((im_h*im_w, 1))
    rgb_pts_b = color_img[:, :, 2].reshape((im_h*im_w, 1))

    # It does not matter what the internal values or xyz points of the cam pts are in relation
    # to the rgb points. The rgb points only relate the index to color. Position is irrelevant.
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
    
    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    
    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
#     print('len(depth_heightmap == -z_bottom) = {}'.format(len(depth_heightmap == -z_bottom)))
#     print('Any nan values in depth_heightmap: {}'.format(np.any(depth_heightmap == np.nan, 1)))

    return color_heightmap, depth_heightmap


def get_rotated_img(img_tensor, theta_in_rad, device='cuda'):
    print(img_tensor.size())
    img = img_tensor
    theta = theta_in_rad
    theta_affine_matrix = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0]])
    theta_affine_matrix = theta_affine_matrix.reshape(2, 3, 1)
    theta_affine_matrix = torch.from_numpy(theta_affine_matrix).permute(2, 0, 1).float()
    affine_grid = F.affine_grid(theta_affine_matrix, size=img.size()).to(device)
    rotated_img = F.grid_sample(img, affine_grid, mode='nearest')
    
    return rotated_img


def get_input_tensors(color_heightmap, depth_heightmap):
    # Apply 2x scale to input heightmaps
    color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
    depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
    assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

    # Add extra padding (to handle rotations inside network)
    diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
    diag_length = np.ceil(diag_length/32)*32
    padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
    color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
    color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
    color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
    color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
    depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

    # Pre-process color image (scale and normalize)
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    input_color_image = color_heightmap_2x.astype(float)/255
    for c in range(3):
        input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

    # Pre-process depth image (normalize)
    image_mean = [0.01, 0.01, 0.01]
    image_std = [0.03, 0.03, 0.03]
    depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
    input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
    for c in range(3):
        input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

    # Construct minibatch of size 1 (b,c,h,w)
    input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
    input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
    input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
    input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
    
    return input_color_data, input_depth_data


def get_prepared_img(img, mode):
    if mode == 'rgb':
        img = img.astype(np.float)/255
        img[img < 0] += 1
        img *= 255
        img = np.fliplr(img)
        img = img.astype(np.uint8)
    elif mode == 'depth':
        img = np.fliplr(img)
        zNear = 0.01
        zFar = 10
        img = img * (zFar - zNear) + zNear
        
    return img


def transform_position_cam_to_global(position):
    cam_pose = np.asarray([[ 1, 0, 0, 0], [ 0, -0.70710679, -0.70710678, 1], [ 0, 0.70710678, -0.70710679, 0.5]])
    position = np.dot(cam_pose[:, :3], position) + cam_pose[:, 3]
    return position
        

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def depth_img_from_bytes(d_img_raw, res):
    deserialized_depth_bytes = np.frombuffer(d_img_raw, dtype=np.float32)
    depth_img = np.reshape(deserialized_depth_bytes, newshape=(res[1], res[0]))

    return depth_img


def rgb_img_from_bytes(rgb_img_raw, res):
    color_img = []
    for i in range(0, len(rgb_img_raw), 24):
        r = np.frombuffer(rgb_img_raw[i:i+8], dtype=np.int8)
        g = np.frombuffer(rgb_img_raw[i+8:i+16], dtype=np.int8)
        b = np.frombuffer(rgb_img_raw[i+16:i+24], dtype=np.int8)
        color_img.append((r, g, b))

    color_img = np.array(color_img).reshape((res[1], res[0], 3))

    return color_img


# def rotate(img):
#     row, col, channel = img.shape
#     angle = np.random.uniform(-15, 15)
#     rotation_point = (row / 2, col / 2)
#     rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)
#     rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))
    
# #     rot_mat = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 45, 1)
# #     warp_rotate_dst = cv2.warpAffine(cv2.UMat(img),rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST).get()

#     return rotated_img 


