#!/usr/bin/env python3

# Built-in lib imports
import time, random, struct, pickle

# Standard Library imports
import numpy as np
import math

# Third-Party imports
import b0RemoteApi
import torch.nn as nn
import torch
from scipy import ndimage


import skimage.transform as trans
# from utils_warp import convert_image_np, normalize_transforms, rotatepoints, show_image
from utils import get_heightmap, get_input_tensors, get_prepared_img, \
    transform_position_cam_to_global, euler2rotm, isRotm, depth_img_from_bytes, \
    rgb_img_from_bytes


from model import reinforcement_net



with b0RemoteApi.RemoteApiClient('b0RemoteApi_V-REP','b0RemoteApi', timeout=5) as client:

    # Make sure simulation is not running
    client.simxStopSimulation(client.simxDefaultPublisher())

    # Global variables
    doNextStep = True
    rgb_vision_msg = None
    d_vision_msg = None


    ##########################
    # Callbacks

    # Callbacks
    def simulationStepStarted(msg):
        # simTime=msg[1][b'simulationTime'];
        # print('Simulation step started. Simulation time: ',simTime)
        pass

    def simulationStepDone(msg):
        global doNextStep
        doNextStep = True

    #######################


    ##########################
    # Robot Class

    class Robot():
        def __init__(self):
            # Create object handles
            _, self.target_right_handle = client.simxGetObjectHandle("UR5_target", client.simxServiceCall())
            _, self.connector_handle = client.simxGetObjectHandle('RG2_attachPoint', client.simxServiceCall())
            _, self.sensor_handle = client.simxGetObjectHandle('RG2_attachProxSensor', client.simxServiceCall())
            _, self.gripper_joint_handle = client.simxGetObjectHandle('RG2_openCloseJoint', client.simxServiceCall())
            _, self.cube_handle = client.simxGetObjectHandle("cube", client.simxServiceCall())
            _, self.right_force_sensor_handle = client.simxGetObjectHandle("RG2_rightForceSensor", client.simxServiceCall())
            _, self.vision_sensor_handle = client.simxGetObjectHandle('vision_sensor', client.simxServiceCall())
            _, self.side_vision_sensor_handle = client.simxGetObjectHandle('side_vision_sensor', client.simxServiceCall())

            print('target_parent: {}'.format(client.simxGetObjectParent(self.target_right_handle, client.simxServiceCall())[1]))

            # not sure about the values
            # self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            self.workspace_limits = np.asarray([[-0.6, -0.3], [-0.175, 0.175], [0.03, 0.3]]) # np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # np.asarray([[-2, 2], [-6, -2], [-5, -4]]) # note workspace limits
            self.heightmap_resolution = 0.002 # No idea # HHHHHHHHHHHHHERRRRRRRRRREEEEEEEEEEEEEEE
            # cam pose in vrep
            # self.cam_pose = np.asarray([[ 1, 0, 0, 0], [ 0, -0.70710679, -0.70710678, 1], [ 0, 0.70710678, -0.70710679, 0.5]])

            # Initiate simulation options
            client.simxSynchronous(False)
            client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
            client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))
            client.simxStartSimulation(client.simxDefaultPublisher())

            client.simxAddStatusbarMessage("Starting!!!", client.simxDefaultPublisher())
            print('Started Simulation!')

            # 
            # self.camera_position = client.simxGetObjectPosition(self.vision_sensor_handle,  -1, client.simxServiceCall())
            # self.camera_orientation = client.simxGetObjectOrientation(self.vision_sensor_handle,  -1, client.simxServiceCall())
            # self.rotation_matrix = euler2rotm(self.camera_orientation[1])
            # self.camera_transformation = client.simxGetObjectMatrix(self.vision_sensor_handle,  -1, client.simxServiceCall())

            self.model = reinforcement_net(use_cuda=False)

            self.setup_sim_camera()


        def setup_sim_camera(self):

            # Get handle to camera
            sim_ret, self.cam_handle = client.simxGetObjectHandle('vision_sensor', client.simxServiceCall())

            # Get camera pose and intrinsics in simulation
            sim_ret, cam_position = client.simxGetObjectPosition(self.cam_handle, -1, client.simxServiceCall())
            sim_ret, cam_orientation = client.simxGetObjectOrientation(self.cam_handle, -1, client.simxServiceCall())
            cam_trans = np.eye(4,4)
            cam_trans[0:3,3] = np.asarray(cam_position)
            cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
            cam_rotm = np.eye(4,4)
            cam_rotm[0:3,0:3] = np.linalg.inv(euler2rotm(cam_orientation))
            self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
            self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            self.cam_depth_scale = 1

            # Get background image
            self.bg_color_img, self.bg_depth_img = self.get_camera_data()
            self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


        # Open Gripper
        def open_gripper(self):
            motor_velocity = 0.5 # m/s
            motor_force = 100 # N
            client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
            client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
            gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
            while gripper_position[1] < 0:

                gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())

                # client.simxSynchronousTrigger()
                # client.simxSpinOnce()

        # Close Gripper
        def close_gripper(self):
            motor_velocity = -0.5 # m/s
            motor_force = 100 # N
            client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
            client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
            gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
            right_force_sensor_feedback= client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
            while gripper_position[1] > -0.046:#  and right_force_sensor_feedback[2][2] > -80:
                gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
                right_force_sensor_feedback = client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
                # print("right_force_sensor_feedback: ", right_force_sensor_feedback[2])

                # client.simxSynchronousTrigger()
                # client.simxSpinOnce()


        def move_to_rand(self):
            random_position = [np.random.randint(limits[0], limits[1]) for limits in self.workspace_limits]
            random_position = [+.5163e-01, +4.4720e-01, +3.8412e-01] # transform_position_cam_to_global(random_position)
            WORLD_FRAME = -1 # vision_sensor_handle
            _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
            print(current_position)
            move_direction = np.asarray([random_position[0] - current_position[0], random_position[1] - current_position[1], random_position[2] - current_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            step = 0.02*(move_direction/move_magnitude) # 0.02
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for i_step in range(num_move_steps):
                next_position = [current_position[0] + step[0], current_position[1] + step[1], current_position[2] + step[2]]
                client.simxSetObjectPosition(self.target_right_handle, WORLD_FRAME, next_position, client.simxServiceCall())
                _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
                # client.simxSynchronousTrigger()
                # client.simxSpinOnce()

            # next_position = [current_position[0] + step[0], current_position[1] + step[1], current_position[2] + step[2]]
            # client.simxSetObjectPosition(target_right_handle, WORLD_FRAME, next_position, client.simxServiceCall())


        def move_to(self, target):
            random_position = target # transform_position_cam_to_global(random_position)
            WORLD_FRAME = 20 # -1 # vision_sensor_handle
            _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
            print('current_position {}'.format(current_position))
            WORLD_FRAME = -1# self.vision_sensor_handle
            _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
            print('current_position {}'.format(current_position))
            
            move_direction = np.asarray([random_position[0] - current_position[0], random_position[1] - current_position[1], random_position[2] - current_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            step = 0.02*(move_direction/move_magnitude) # 0.02
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for i_step in range(num_move_steps):
                next_position = [current_position[0] + step[0], current_position[1] + step[1], current_position[2] + step[2]]
                client.simxSetObjectPosition(self.target_right_handle, WORLD_FRAME, next_position, client.simxServiceCall())
                _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
                # client.simxSynchronousTrigger()
                # client.simxSpinOnce()

            # next_position = [current_position[0] + step[0], current_position[1] + step[1], current_position[2] + step[2]]
            # client.simxSetObjectPosition(target_right_handle, WORLD_FRAME, next_position, client.simxServiceCall())



        def get_camera_data(self):

            start_time = time.time()
            rgb_ret, rgb_res, rgb_img_raw = client.simxGetVisionSensorImage(self.vision_sensor_handle, False, client.simxServiceCall())
            end_time = time.time()
            # print('Elapsed rgb capture: {}'.format(end_time-start_time))

            start_time = time.time()
            d_ret, d_res, d_img_raw = client.simxGetVisionSensorDepthBuffer(self.vision_sensor_handle, True, True, client.simxServiceCall())
            end_time = time.time()
            # print('Elapsed depth capture: {}'.format(end_time-start_time))

            assert (d_ret and rgb_ret) == True

            client.simxSetVisionSensorImage(self.side_vision_sensor_handle, False, rgb_img_raw, client.simxDefaultPublisher())
            
            depth_img = depth_img_from_bytes(d_img_raw, rgb_res)
            # np.save('depth_img', depth_img)

            color_img = rgb_img_from_bytes(rgb_img_raw, rgb_res)
            # np.save('color_img', color_img)

            return color_img, depth_img

        def get_input_color_and_depth_data(self):
            color_img, depth_img = self.get_camera_data()
            color_img = get_prepared_img(color_img, 'rgb')
            depth_img = get_prepared_img(depth_img, 'depth')
            color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, robot.workspace_limits, robot.heightmap_resolution)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
            input_color_data, input_depth_data = get_input_tensors(color_heightmap, valid_depth_heightmap)

            return input_color_data, input_depth_data


        def get_action(self):

            color_img, depth_img = self.get_camera_data()
            print('color_img.shape: {}'.format(color_img.shape))
            color_img = get_prepared_img(color_img, 'rgb')
            depth_img = get_prepared_img(depth_img, 'depth')
            color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, robot.workspace_limits, robot.heightmap_resolution)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
            depth_heightmap = valid_depth_heightmap
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

            # Pass input data through model
            output_prob, state_feat = self.model.forward(input_color_data, input_depth_data) # is_volatile, specific_rotation)


            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

            print('grasp_predictions.shape: {}'.format(grasp_predictions.shape))
            # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            predicted_value = np.max(grasp_predictions)

            # Compute 3D position of pixel
            print('Action: %s at (%d, %d, %d)' % ('Grasp', best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))
            best_rotation_angle = np.deg2rad(best_pix_ind[0]*(360.0/robot.model.num_rotations))
            best_pix_x = best_pix_ind[2]
            best_pix_y = best_pix_ind[1]
            primitive_position = [best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0], best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]]

            return primitive_position # grasp_predictions, state_feat

######################################################################################################
######################################################################################################
######################################## Main ########################################################
######################################################################################################
######################################################################################################
#     robot = Robot()
#     startTime=time.time()
#     # while time.time()<startTime+5: 
#     if doNextStep:
#         doNextStep=False
#         input_color_data, input_depth_data = robot.get_input_color_and_depth_data()
#         out = robot.model.forward(input_color_data.cpu(), input_depth_data.cpu())
#         robot.move_to_rand()
#         client.simxSleep(0.5)
#         robot.open_gripper()
#         client.simxSleep(0.5)
#         robot.close_gripper()
#         # client.simxSynchronousTrigger()
#     client.simxSpinOnce()
# client.simxStopSimulation(client.simxDefaultPublisher())

# # Stop simulation
# print('Stopping Simulation...')
# client.simxStopSimulation(client.simxDefaultPublisher())
# print('Simulation Over.')


    robot = Robot()
    primitive_position = robot.get_action()
    print(primitive_position)

    # robot.move_to_rand()
    robot.move_to(primitive_position)
    client.simxSleep(0.5)
    robot.open_gripper()
    client.simxSleep(0.5)
    robot.close_gripper()

    # Stop simulation
    print('Stopping Simulation...')
    client.simxStopSimulation(client.simxServiceCall())
    print('Simulation Over.')

