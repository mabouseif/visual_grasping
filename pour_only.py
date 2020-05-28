#!/usr/bin/env python3

# Built-in lib imports
import time, random, struct, pickle

# Standard Library imports
import numpy as np
import math
import matplotlib.pyplot as plt

# Third-Party imports
import b0RemoteApi
import torch.nn as nn
import torch
import torch.optim as optim
from scipy import ndimage
import torch
from torch.autograd import Variable
import cv2
import os
import argparse

import skimage.transform as trans
# from utils_warp import convert_image_np, normalize_transforms, rotatepoints, show_image
from utils import get_heightmap, get_input_tensors, get_prepared_img, \
    transform_position_cam_to_global, euler2rotm, isRotm, depth_img_from_bytes, \
    rgb_img_from_bytes


from model import reinforcement_net
import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    args = parser.parse_args()
    TEST = args.test

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
            def __init__(self, USE_CUDA=False, LATERAL=False, IS_TESTING=False):
                # Create object handles
                _, self.target_right_handle = client.simxGetObjectHandle("UR5_target", client.simxServiceCall())
                _, self.cube_handle = client.simxGetObjectHandle("obj_cube", client.simxServiceCall())
                _, self.vision_sensor_handle = client.simxGetObjectHandle('vision_sensor', client.simxServiceCall())
                _, self.side_vision_sensor_handle = client.simxGetObjectHandle('side_vision_sensor', client.simxServiceCall())
                _, self.cup_handle = client.simxGetObjectHandle('Cup', client.simxServiceCall())
                _, self.sphere_fluid_handle = client.simxGetObjectHandle('sphere_fluid', client.simxServiceCall())
                _, self.bowl_handle = client.simxGetObjectHandle('Bowl', client.simxServiceCall())


                _, self.j0=client.simxGetObjectHandle("MicoHand_fingers12_motor1", client.simxServiceCall())
                _, self.j1=client.simxGetObjectHandle("MicoHand_fingers12_motor2", client.simxServiceCall())
                _, self.f1_bottom = client.simxGetObjectHandle("MicoHand_forceSens1_finger1", client.simxServiceCall())
                _, self.f1_top = client.simxGetObjectHandle("MicoHand_forceSens2_finger1", client.simxServiceCall())
                _, self.f2_bottom = client.simxGetObjectHandle("MicoHand_forceSens1_finger3", client.simxServiceCall())
                _, self.f2_top = client.simxGetObjectHandle("MicoHand_forceSens2_finger3", client.simxServiceCall())

                # Parameters
                self.use_cuda = USE_CUDA
                self.lateral = LATERAL
                self.is_testing = IS_TESTING
                self.explore_prob = 0.1 if self.is_testing else 0.5
                self.learning_rate = 1e-4
                self.future_reward_discount = 0.5
                self.iteration = 1
                self.explore_rate_decay = True
                self.experience_replay = True
                self.label_value_log = []
                self.reward_value_log  = []
                self.predicted_value_log = []
                self.executed_action_log = []
                _, self.original_bowl_position = client.simxGetObjectPosition(self.bowl_handle, -1, client.simxServiceCall())
                _, self.original_cup_position = client.simxGetObjectPosition(self.cup_handle, -1, client.simxServiceCall())
                self.obj_list = ["obj_cube", "obj_cube_large", "obj_cylinder", "obj_cylinder_small", "obj_cuboid", "obj_cuboid_long"] # "obj_cuboid_thin"
                self.object_handles = [client.simxGetObjectHandle(obj, client.simxServiceCall())[1] for obj in self.obj_list]
                self.obj_count = len(self.object_handles)

                # Initialize Huber loss
                self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
                if self.use_cuda:
                    self.criterion = self.criterion.cuda()  

                # Q-Network
                rotations = 1 if self.lateral else 16
                self.model = reinforcement_net(use_cuda=self.use_cuda, rotations=rotations)
                if self.is_testing:
                    self.model.load_state_dict(torch.load('/home/mohamed/drive/coppelia_stuff/scripts/logs/mico_pour_1/models/snapshot-000899.grasp.pth'))
                if self.use_cuda:
                    self.model = self.model.cuda()


                # Initialize optimizer
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)

                self.logger = logger.Logger(False, './logs')

                # not sure about the values
                # self.workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [0.0001, 0.4]])
                self.workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [0.2, 0.4]])
                self.heightmap_resolution = 0.002 # No idea # HHHHHHHHHHHHHERRRRRRRRRREEEEEEEEEEEEEEE

                # Initiate simulation options
                client.simxSynchronous(False)
                client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
                client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))
                client.simxStartSimulation(client.simxDefaultPublisher())

                client.simxAddStatusbarMessage("Starting!!!", client.simxDefaultPublisher())
                print('Started Simulation!')


                self.setup_sim_camera()

                if not self.lateral:
                    self.reset_objects()


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
                # motor_force = 100 # N
                # client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
                client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
                _, gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
                right_force_sensor_feedback= client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
                gripper_fully_closed = False
                while gripper_position > -0.046:#  and right_force_sensor_feedback[2][2] > -80:
                    _, new_gripper_joint_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
                    # right_force_sensor_feedback = client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
                    if new_gripper_joint_position >= gripper_position:
                        return gripper_fully_closed
                    gripper_position = new_gripper_joint_position

                gripper_fully_closed = True

                return gripper_fully_closed


            def move_to_rand(self):
                random_position = [np.random.randint(limits[0], limits[1]) for limits in self.workspace_limits]
                random_position = [+.5163e-01, +4.4720e-01, +3.8412e-01] # transform_position_cam_to_global(random_position)
                WORLD_FRAME = -1 # vision_sensor_handle
                _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
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
                WORLD_FRAME = -1# self.vision_sensor_handle
                _, current_position = client.simxGetObjectPosition(self.target_right_handle, WORLD_FRAME, client.simxServiceCall())
                
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
                d_ret, d_res, d_img_raw = client.simxGetVisionSensorDepthBuffer(self.vision_sensor_handle, False, True, client.simxServiceCall())
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



            def reset_cube(self):
                relObjHandle = -1
                # position = [-5.1500e-01, -1.5000e-02, +1.5000e-02]
                position = [np.random.uniform(self.workspace_limits[0][0], self.workspace_limits[0][1]), np.random.uniform(self.workspace_limits[1][0], self.workspace_limits[1][1]), +3.0000e-02]
                orientation = [0, 0, 0]
                ret_pos = client.simxSetObjectPosition(self.cube_handle, relObjHandle, position, client.simxServiceCall())
                ret_orient = client.simxSetObjectOrientation(self.cube_handle, relObjHandle, orientation, client.simxServiceCall())
                if not (ret_pos and ret_orient):
                    print('Failed to set cube back to position')
                    exit


            def reset_cup(self):
                # sim_ret, object_position = client.simxGetObjectPosition(self.cup_handle, -1, client.simxServiceCall())
                # if ((object_position[0] < self.workspace_limits[0][0]) or (object_position[0] > self.workspace_limits[0][1])) or \
                #     ((object_position[1] < self.workspace_limits[1][0]) or (object_position[1] > self.workspace_limits[1][1])) or \
                #         ((object_position[2] < self.workspace_limits[2][0]) or (object_position[2] > self.workspace_limits[2][1])):
                relObjHandle = -1
                position = [self.original_cup_position[0], self.original_cup_position[1], self.original_cup_position[2]] # [self.original_cup_position[0] + np.random.uniform(-0.05, 0.05), self.original_cup_position[1] + np.random.uniform(-0.1, 0.1), self.original_cup_position[2]]
                orientation = [0, 0, 0]# [np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), 0]
                ret_pos = client.simxSetObjectPosition(self.cup_handle, relObjHandle, position, client.simxServiceCall())
                ret_orient = client.simxSetObjectOrientation(self.cup_handle, relObjHandle, orientation, client.simxServiceCall())

                client.simxSetObjectPosition(self.sphere_fluid_handle, relObjHandle, position, client.simxServiceCall())
                


            def reset_fluid(self):
                relObjHandle = -1
                _, ret_pos = client.simxGetObjectPosition(self.cup_handle, relObjHandle, client.simxServiceCall())
                client.simxSetObjectPosition(self.sphere_fluid_handle, relObjHandle, ret_pos, client.simxServiceCall())

            def reset_bowl(self):
                relObjHandle = -1
                position = [self.original_bowl_position[0] + np.random.uniform(-0.5, 0.5), self.original_bowl_position[1] + np.random.uniform(-0.5, 0.5), self.original_bowl_position[2]]
                client.simxSetObjectPosition(self.bowl_handle, relObjHandle, position, client.simxServiceCall())
                     


            def reset_objects(self):
                # Relative frame handle
                relObjHandle = -1
                # Get all shape object handles
                # obj_list = ["obj_cube", "obj_cube_large", "obj_cylinder", "obj_cylinder_small", "obj_cuboid", "obj_cuboid_thin", "obj_cuboid_long"]
                # obj_handles = [client.simxGetObjectHandle(obj_name, client.simxServiceCall())[1] for obj_name in obj_list]
                # Filter object handles by names starting with "obj"
                for handle in self.object_handles:
                    position = [np.random.uniform(self.workspace_limits[0][0], self.workspace_limits[0][1]), np.random.uniform(self.workspace_limits[1][0], self.workspace_limits[1][1]), +3.0000e-02]
                    orientation = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), 0]
                    ret_pos = client.simxSetObjectPosition(handle, relObjHandle, position, client.simxServiceCall())
                    ret_orient = client.simxSetObjectOrientation(handle, relObjHandle, orientation, client.simxServiceCall())


            def get_obj_positions(self):
                obj_positions = []
                for object_handle in self.object_handles:
                    sim_ret, object_position = client.simxGetObjectPosition(object_handle, -1, client.simxServiceCall())
                    obj_positions.append(object_position)
                return obj_positions
            
                # Move the grasped object elsewhere
            def remove_grasped_object(self):
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:,2]
                grasped_object_ind = np.argmax(object_positions)
                grasped_object_handle = self.object_handles[grasped_object_ind]
                client.simxSetObjectPosition(grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1), client.simxServiceCall())
                self.obj_count -= 1
                    


            def workspace_empty(self):
                obj_out = []
                for object_handle in self.object_handles:
                    sim_ret, object_position = client.simxGetObjectPosition(object_handle, -1, client.simxServiceCall())
                    if ((object_position[0] < self.workspace_limits[0][0]) or (object_position[0] > self.workspace_limits[0][1])) or \
                        ((object_position[1] < self.workspace_limits[1][0]) or (object_position[1] > self.workspace_limits[1][1])) or \
                            ((object_position[2] < self.workspace_limits[2][0]) or (object_position[2] > self.workspace_limits[2][1])):
                        obj_out.append(True)

                if len(obj_out) == len(self.object_handles):
                    return True

                return False



            def save_snapshot(self, iteration):
                self.logger.save_model(iteration, self.model, 'grasp')
                if self.use_cuda:
                    self.model = self.model.cuda()
                print('-'*50)
                print('Model saved.')
                print('-'*50)



            def close_mico(self):
                closingVel=0.1
                motor_force = 120 # N
                client.simxSetJointForce(self.j0, motor_force, client.simxServiceCall())
                client.simxSetJointForce(self.j1, motor_force, client.simxServiceCall())
                client.simxSetJointTargetVelocity(self.j0, -closingVel, client.simxServiceCall())
                client.simxSetJointTargetVelocity(self.j1, -closingVel, client.simxServiceCall())

                _, j0_position = client.simxGetJointPosition(self.j0, client.simxServiceCall())
                _, j1_position = client.simxGetJointPosition(self.j1, client.simxServiceCall())

                f1_bottom= client.simxReadForceSensor(self.f1_bottom, client.simxServiceCall())
                f1_top = client.simxReadForceSensor(self.f1_top, client.simxServiceCall())
                f2_bottom= client.simxReadForceSensor(self.f2_bottom, client.simxServiceCall())
                f2_top= client.simxReadForceSensor(self.f2_top, client.simxServiceCall())
                gripper_fully_closed = False
                closing_position = -0.04# -4.462e-02 # starting -2.742e-06
                while (j0_position > closing_position) or (j1_position > closing_position): #  and right_force_sensor_feedback[2][2] > -80:
                    _, new_j0_position = client.simxGetJointPosition(self.j0, client.simxServiceCall())
                    _, new_j1_position = client.simxGetJointPosition(self.j1, client.simxServiceCall())
                    # _, new_j2_position = client.simxGetJointPosition(self.j2, client.simxServiceCall())
                    # _, new_j3_position = client.simxGetJointPosition(self.j3, client.simxServiceCall())
                    f1_bottom= client.simxReadForceSensor(self.f1_bottom, client.simxServiceCall())
                    f1_top = client.simxReadForceSensor(self.f1_top, client.simxServiceCall())
                    f2_bottom= client.simxReadForceSensor(self.f2_bottom, client.simxServiceCall())
                    f2_top= client.simxReadForceSensor(self.f2_top, client.simxServiceCall())
                    print('+'*50)
                    print("f1_bottom: {}".format(f1_bottom[2]))
                    print("f1_top: {}".format(f1_top[2]))
                    print("f2_bottom: {}".format(f2_bottom[2]))
                    print("f2_top: {}".format(f2_top[2]))
                    print('+'*50)
                    if (f1_top[2][2] > 20) and (f2_top[2][2] > 20):
                        return gripper_fully_closed
                    if (new_j0_position >= j0_position) or (new_j1_position >= j1_position): # If no change, AND not reached closing position, then gripper is not fully closed
                        break

                    j0_position = new_j0_position
                    j1_position = new_j1_position
                    # j2_position = new_j2_position
                    # j3_position = new_j3_position

                gripper_fully_closed = True

                return gripper_fully_closed


            def open_mico(self):
                closingVel=0.1
                client.simxSetJointTargetVelocity(self.j0, closingVel, client.simxServiceCall())
                client.simxSetJointTargetVelocity(self.j1, closingVel, client.simxServiceCall())

                # motor_velocity = 0.5 # m/s
                # motor_force = 100 # N
                # client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
                # client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
                gripper_position = client.simxGetJointPosition(self.j0, client.simxServiceCall())
                while gripper_position[1] < 0:
                    gripper_position = client.simxGetJointPosition(self.j0, client.simxServiceCall())


            def pour(self, position):
                # Initialize variables that influence reward
                grasp_success = False

                # Compute tool orientation from heightmap rotation angle
                tool_rotation_angle = np.pi

                # Avoid collision with table
                position = np.asarray(position).copy()
                # height_z = 0.05 # 0.04
                # position[2] = max(position[2] - 0.2, self.workspace_limits[2][0] + 0.02)

                # Move gripper to location above grasp target
                grasp_location_margin = 0.1
                # sim_ret, UR5_target_handle = client.simxGetObjectHandle(self.sim_client,'UR5_target',client.simx_opmode_blocking)
                location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

                # # Compute gripper position and linear movement increments
                tool_position = location_above_grasp_target
                # self.move_to(tool_position)

                # Compute gripper position and linear movement increments
                tool_position = location_above_grasp_target
                sim_ret, UR5_target_position = client.simxGetObjectPosition(self.target_right_handle, -1, client.simxServiceCall())
                move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.05*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_direction[0]/move_step[0]))


                # Compute gripper orientation and rotation increments
                sim_ret, gripper_orientation = client.simxGetObjectOrientation(self.target_right_handle, -1, client.simxServiceCall())
                rotation_step = 0.05 if (tool_rotation_angle - gripper_orientation[0] > 0) else -0.05
                num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[0])/rotation_step))

                # Simultaneously move and rotate gripper
                for step_iter in range(max(num_move_steps, 0)):
                    client.simxSetObjectPosition(self.target_right_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),client.simxServiceCall())
                    # client.simxSetObjectOrientation(self.target_right_handle, -1, (gripper_orientation[0] + rotation_step*min(step_iter,num_rotation_steps), gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())
                client.simxSetObjectPosition(self.target_right_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),client.simxServiceCall())
                # client.simxSetObjectOrientation(self.target_right_handle, -1, (tool_rotation_angle, gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())

                # Approach grasp target
                # self.move_to(position)

                # Simultaneously move and rotate gripper
                for step_iter in range(max(0, num_rotation_steps)):
                    # client.simxSetObjectPosition(self.target_right_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),client.simxServiceCall())
                    client.simxSetObjectOrientation(self.target_right_handle, -1, (gripper_orientation[0] + rotation_step*min(step_iter,num_rotation_steps), gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())
                # client.simxSetObjectPosition(self.target_right_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),client.simxServiceCall())
                client.simxSetObjectOrientation(self.target_right_handle, -1, (tool_rotation_angle, gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())

                client.simxSleep(0.75)

                # Simultaneously move and rotate gripper
                sim_ret, gripper_orientation = client.simxGetObjectOrientation(self.target_right_handle, -1, client.simxServiceCall())
                for step_iter in range(max(0, num_rotation_steps)):
                    # client.simxSetObjectPosition(self.target_right_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),client.simxServiceCall())
                    client.simxSetObjectOrientation(self.target_right_handle, -1, (gripper_orientation[0] - rotation_step*min(step_iter,num_rotation_steps), gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())
                # client.simxSetObjectPosition(self.target_right_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),client.simxServiceCall())
                client.simxSetObjectOrientation(self.target_right_handle, -1, (np.pi/2, gripper_orientation[1], gripper_orientation[2]), client.simxServiceCall())


                _, cup_position = client.simxGetObjectPosition(self.cup_handle, -1, client.simxServiceCall())
                _, sphere_fluid_position = client.simxGetObjectPosition(self.sphere_fluid_handle, -1, client.simxServiceCall())

                pour_success = self.bowl_full()

                print('Pour Sucess: {}'.format(pour_success))


                # # Close gripper to grasp target
                # gripper_full_closed = self.close_mico()

                # # Move gripper to location above grasp target
                # location_above_grasp_target = position
                # location_above_grasp_target[2] = location_above_grasp_target[2] + 0.1
                # location_above_grasp_target[0] = location_above_grasp_target[0] + grasp_location_margin
                # self.move_to(location_above_grasp_target)

                # # Check if grasp is successful
                # gripper_full_closed = self.close_mico()
                # grasp_success = not gripper_full_closed

                # print('Grasp success: {}'.format(grasp_success))

                # self.open_mico()

                self.move_to([-3.4349e-01, +3.2925e-02, 0.25])

                return pour_success



            def grasp_lateral(self, position, best_rotation_angle):

                # Initialize variables that influence reward
                grasp_success = False
                change_detected = False

                # Compute tool orientation from heightmap rotation angle
                tool_rotation_angle = np.pi/2

                # Avoid collision with table
                position = np.asarray(position).copy()
                height_z = 0.05 # 0.04
                position[2] = max(position[2] - 0.2, self.workspace_limits[2][0] + 0.02)

                # Move gripper to location above grasp target
                grasp_location_margin = 0.15
                # sim_ret, UR5_target_handle = client.simxGetObjectHandle(self.sim_client,'UR5_target',client.simx_opmode_blocking)
                location_above_grasp_target = (position[0] + grasp_location_margin, position[1], position[2])

                # # Compute gripper position and linear movement increments
                tool_position = location_above_grasp_target
                # self.move_to(tool_position)

                # Compute gripper position and linear movement increments
                tool_position = location_above_grasp_target
                sim_ret, UR5_target_position = client.simxGetObjectPosition(self.target_right_handle, -1, client.simxServiceCall())
                move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.05*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_direction[0]/move_step[0]))


                # Compute gripper orientation and rotation increments
                sim_ret, gripper_orientation = client.simxGetObjectOrientation(self.target_right_handle, -1, client.simxServiceCall())
                rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[2] > 0) else -0.3
                num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[2])/rotation_step))

                # Simultaneously move and rotate gripper
                for step_iter in range(max(num_move_steps, num_rotation_steps)):
                    client.simxSetObjectPosition(self.target_right_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),client.simxServiceCall())
                    # client.simxSetObjectOrientation(self.target_right_handle, -1, (gripper_orientation[0], gripper_orientation[1], gripper_orientation[2] + rotation_step*min(step_iter,num_rotation_steps)), client.simxServiceCall())
                client.simxSetObjectPosition(self.target_right_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),client.simxServiceCall())
                # client.simxSetObjectOrientation(self.target_right_handle, -1, (gripper_orientation[0], gripper_orientation[1], tool_rotation_angle), client.simxServiceCall())

                # Ensure gripper is open
                self.open_mico()

                # Approach grasp target
                self.move_to(position)

                # Close gripper to grasp target
                gripper_full_closed = self.close_mico()

                # Move gripper to location above grasp target
                location_above_grasp_target = position
                location_above_grasp_target[2] = location_above_grasp_target[2] + 0.1
                location_above_grasp_target[0] = location_above_grasp_target[0] + grasp_location_margin
                self.move_to(location_above_grasp_target)

                # Check if grasp is successful
                gripper_full_closed = self.close_mico()
                grasp_success = not gripper_full_closed

                print('Grasp success: {}'.format(grasp_success))

                # self.open_mico()

                self.move_to([-3.4349e-01, +3.2925e-02, 0.25])

                return grasp_success



            def bowl_full(self):
                thresh = 0.1 # 0.5
                _, fluid_position = client.simxGetObjectPosition(self.sphere_fluid_handle, -1, client.simxServiceCall())
                if np.sqrt((fluid_position[0] - self.original_bowl_position[0])**2 + \
                    (fluid_position[1] - self.original_bowl_position[1])**2 + \
                        (fluid_position[2] - self.original_bowl_position[2])**2) < thresh:
                    return True
                return False


            def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

                best = False
                canvas = None
                prediction_vis = predictions[0,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (255,255,255), 2) # (0,0,255)
                background_image = ndimage.rotate(color_heightmap, 0, reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)

                return prediction_vis


            def trainer_forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
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
                output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile=is_volatile) # is_volatile, specific_rotation)
                print("output_prob.shape: {}".format(output_prob[0][0].size()))
                print((output_prob[0][0].size()[-1]))
                self.output_prob_shape = output_prob[0][0].size()[-1]

                # Return Q values (and remove extra padding)
                for rotate_idx in range(len(output_prob)):
                    if rotate_idx == 0:
                        grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    else:
                        grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0) 

                return grasp_predictions, state_feat

            # Compute labels and backpropagate
            def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value): 

                output_prob_dim = self.output_prob_shape # 288
                heightmap_dim = int(np.round(((self.workspace_limits[0][1] - self.workspace_limits[0][0])*1000)/2)) # 200
                print("heightmap_dim: {}".format(heightmap_dim))
                # print(((self.workspace_limits[0][1] - self.workspace_limits[0][0])*1000)/2)
                starting_pixel = int((output_prob_dim - heightmap_dim) / 2)

                # Compute labels
                label = np.zeros((1,output_prob_dim,output_prob_dim))
                action_area = np.zeros((heightmap_dim,heightmap_dim)) # np.zeros((224,224))
                action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
                # blur_kernel = np.ones((5,5),np.float32)/25
                # action_area = cv2.filter2D(action_area, -1, blur_kernel)
                tmp_label = np.zeros((heightmap_dim,heightmap_dim))
                tmp_label[action_area > 0] = label_value
                label[0,starting_pixel:(output_prob_dim-starting_pixel),starting_pixel:(output_prob_dim-starting_pixel)] = tmp_label

                # Compute label mask
                label_weights = np.zeros(label.shape)
                tmp_label_weights = np.zeros((heightmap_dim,heightmap_dim))
                tmp_label_weights[action_area > 0] = 1
                label_weights[0,starting_pixel:(output_prob_dim-starting_pixel),starting_pixel:(output_prob_dim-starting_pixel)] = tmp_label_weights

                # Compute loss and backward pass
                self.optimizer.zero_grad()
                loss_value = 0

                # Do forward pass with specified rotation (to save gradients)
                grasp_predictions, state_feat = self.trainer_forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,output_prob_dim,output_prob_dim), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,output_prob_dim,output_prob_dim), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                grasp_predictions, state_feat = self.trainer_forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,output_prob_dim,output_prob_dim), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,output_prob_dim,output_prob_dim), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

                loss = loss.sum()                   
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

                print('Training loss: %f' % (loss_value))
                self.optimizer.step()


            def get_label_value(self, primitive_action, grasp_success, change_detected, prev_grasp_predictions, next_color_heightmap, next_depth_heightmap):

                # Compute current reward
                current_reward = 0
                if grasp_success:
                    current_reward = 1.0
                # elif change_detected:
                #     current_reward = 0.5

                change_detected = False

                # Compute future reward
                if not change_detected and not grasp_success:
                    future_reward = 0
                else:
                    next_grasp_predictions, next_state_feat = self.trainer_forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                    future_reward = np.max(next_grasp_predictions)

                    # # Experiment: use Q differences
                    # push_predictions_difference = next_push_predictions - prev_push_predictions
                    # grasp_predictions_difference = next_grasp_predictions - prev_grasp_predictions
                    # future_reward = max(np.max(push_predictions_difference), np.max(grasp_predictions_difference))

                print('Current reward: %f' % (current_reward))
                print('Future reward: %f' % (future_reward))
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
                
                return expected_reward, current_reward




            def get_action(self, iteration):

                self.execute_action = True

                for i in range(2):
                    color_img, depth_img = self.get_camera_data()
                    color_img = get_prepared_img(color_img, 'rgb')
                    depth_img = get_prepared_img(depth_img, 'depth')
                    color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, robot.workspace_limits, robot.heightmap_resolution)
                    valid_depth_heightmap = depth_heightmap.copy()
                    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

                    # Save RGB-D images and RGB-D heightmaps # trainer.iteration
                    self.logger.save_images(iteration, color_img, depth_img, '0') # trainer.iteration
                    self.logger.save_heightmaps(iteration, color_heightmap, valid_depth_heightmap, '0') # trainer.iteration


                    grasp_predictions, state_feat = self.trainer_forward(color_heightmap, valid_depth_heightmap, is_volatile=True)


                    ############################################ EXECUTING THREAD ############################################
                    ############################################ EXECUTING THREAD ############################################
                    ############################################ EXECUTING THREAD ############################################

                    if self.execute_action:

                        explore_actions = np.random.uniform() < self.explore_prob
                        if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                            print('Strategy: explore (exploration probability: %f)' % (self.explore_prob))
                            # best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                            best_pix_ind = [np.random.randint(0, grasp_predictions.shape[0]), np.random.randint(0, grasp_predictions.shape[1]), np.random.randint(0, grasp_predictions.shape[2])]
                            predicted_value = grasp_predictions[best_pix_ind[0]][best_pix_ind[1]][best_pix_ind[2]]
                        else:
                            print('Strategy: exploit (exploration probability: %f)' % (self.explore_prob))
                            print('grasp_predictions.shape: {}'.format(grasp_predictions.shape))
                            # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                            predicted_value = np.max(grasp_predictions)

                        # Save predicted confidence value
                        self.predicted_value_log.append([predicted_value])
                        self.logger.write_to_log('predicted-value', self.predicted_value_log)
                            
                        print('best_pix_ind[0]: {}, best_pix_ind[1]: {}, best_pix_ind[2]: {}'.format(best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))

                        # Compute 3D position of pixel
                        print('Action: %s at (%d, %d, %d)' % ('Grasp', best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))
                        best_rotation_angle = np.deg2rad(best_pix_ind[0]*(360.0/robot.model.num_rotations))
                        best_pix_x = best_pix_ind[2] # 0
                        best_pix_y = best_pix_ind[1] # 90
                        primitive_position = [best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0], best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]]
                        position = primitive_position

                        # Save executed primitive
                        self.executed_action_log.append([1, best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]]) # 1 - grasp
                        self.logger.write_to_log('executed-action', self.executed_action_log)



                        # # Visualize executed primitive, and affordances
                        # grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_pix_ind)
                        # logger.save_visualizations(self.iteration, grasp_pred_vis, 'pour')
                        # cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                        print('position: {}'.format(position))
                        # position = [-5.5080e-01, +3.2500e-01, +3.1300e-01]

                        # Execute Primitive
                        grasp_success = self.pour(position)

                        self.execute_action = False


                        if i == 0 and grasp_success:
                            grasp_pred_vis = self.get_prediction_vis(grasp_predictions, color_heightmap, best_pix_ind)
                            self.logger.save_visualizations(iteration, grasp_pred_vis, 'grasp')
                            # self.logger.save_visualizations_best(iteration, best, 'best')
                            cv2.imwrite('visualization.grasp.png', grasp_pred_vis)


                    ########################## TRAINING ##########################
                    ########################## TRAINING ##########################
                    ########################## TRAINING ##########################

                    # Run training iteration in current thread (aka training thread)
                    if 'prev_color_img' in locals() and not self.is_testing:

                        # Detect changes
                        depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                        depth_diff[np.isnan(depth_diff)] = 0
                        depth_diff[depth_diff > 0.3] = 0
                        depth_diff[depth_diff < 0.01] = 0
                        depth_diff[depth_diff > 0] = 1
                        change_threshold = 300
                        change_value = np.sum(depth_diff)
                        change_detected = change_value > change_threshold or prev_grasp_success
                        print('Change detected: %r (value: %d)' % (change_detected, change_value))

                        # if change_detected:
                        #     if prev_primitive_action == 'push':
                        #         no_change_count[0] = 0
                        #     elif prev_primitive_action == 'grasp':
                        #         no_change_count[1] = 0
                        # else:
                        #     if prev_primitive_action == 'push':
                        #         no_change_count[0] += 1
                        #     elif prev_primitive_action == 'grasp':
                        #         no_change_count[1] += 1

                        # Compute training labels
                        label_value, prev_reward_value = self.get_label_value(prev_primitive_action, 
                                                                                prev_grasp_success, 
                                                                                change_detected, 
                                                                                prev_grasp_predictions, 
                                                                                color_heightmap,  # Fix get_label since it's using the local_network call, instead of the trainer call like in the original code, which goes through the preprocessing step.
                                                                                valid_depth_heightmap)
                        
                        self.label_value_log.append([label_value])
                        self.logger.write_to_log('label-value', self.label_value_log)
                        self.reward_value_log.append([prev_reward_value])
                        self.logger.write_to_log('reward-value', self.reward_value_log)

                        # Backpropagate
                        self.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)

                        # self.explore_prob = max(0.5 * np.power(0.9998, iteration),0.1) if self.explore_rate_decay else 0.5
                        self.explore_prob = max(0.5 * np.power(0.999, iteration),0.1) if self.explore_rate_decay else 0.5


                        # Do sampling for experience replay
                        if self.experience_replay:
                            sample_primitive_action = prev_primitive_action
                            sample_primitive_action_id = 1
                            sample_reward_value = 0 if prev_reward_value == 1.0 else 1.0

                            # Get samples of the same primitive but with different results
                            # Indices where the primitive is the prev_prev, and also have different results. This has the same shape as trainer.reward_value_log as well as trainer.executed_action_log.
                            # argwhere returns the indices of the True booleans from the preceding operation.
                            sample_ind = np.argwhere(np.logical_and(np.asarray(self.reward_value_log)[1:iteration,0] == sample_reward_value, np.asarray(self.executed_action_log)[1:iteration,0] == sample_primitive_action_id))

                            if sample_ind.size > 0:
                                # Find sample with highest surprise value
                                sample_surprise_values = np.abs(np.asarray(self.predicted_value_log)[sample_ind[:,0]] - np.asarray(self.label_value_log)[sample_ind[:,0]])
                                sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                                sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                                pow_law_exp = 2
                                rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                                sample_iteration = sorted_sample_ind[rand_sample_ind]
                                print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                                # Load sample RGB-D heightmap
                                print(os.path.join(self.logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                                sample_color_heightmap = cv2.imread(os.path.join(self.logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                                sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                                sample_depth_heightmap = cv2.imread(os.path.join(self.logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                                sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                                # Compute forward pass with sample
                                with torch.no_grad():
                                    sample_grasp_predictions, sample_state_feat = self.trainer_forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

                                # Load next sample RGB-D heightmap
                                next_sample_color_heightmap = cv2.imread(os.path.join(self.logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                                next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                                next_sample_depth_heightmap = cv2.imread(os.path.join(self.logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
                                next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000

                                sample_grasp_success = sample_reward_value == 1
                                # sample_change_detected = sample_push_success
                                # new_sample_label_value, _ = trainer.get_label_value(sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap)

                                # Get labels for sample and backpropagate
                                sample_best_pix_ind = (np.asarray(self.executed_action_log)[sample_iteration,1:4]).astype(int)
                                self.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, self.label_value_log[sample_iteration])

                                # Recompute prediction value and label for replay buffer
                                self.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                                # trainer.label_value_log[sample_iteration] = [new_sample_label_value]

                            else:
                                print('Not enough prior training samples. Skipping experience replay.')


                        
                        # # self.reset_cube()
                        # if grasp_success:
                        #     # self.remove_grasped_object() # or prev_grasp_success?
                        # if self.workspace_empty() and not self.lateral:
                        #     # self.reset_objects()

                    # self.reset_bowl()
                    self.reset_fluid()

                    if self.is_testing:
                        self.execute_action = True
                        # self.reset_bowl()
                        self.reset_fluid()

                    # Save information for next training step
                    prev_color_img = color_img.copy()
                    prev_depth_img = depth_img.copy()
                    prev_color_heightmap = color_heightmap.copy()
                    prev_depth_heightmap = depth_heightmap.copy()
                    prev_valid_depth_heightmap = valid_depth_heightmap.copy()
                    prev_grasp_predictions = grasp_predictions.copy()
                    prev_grasp_success = grasp_success
                    prev_primitive_action = 'grasp'
                    prev_best_pix_ind = best_pix_ind
                    self.iteration += 1


    ######################################################################################################
    ######################################################################################################
    ######################################## Main ########################################################
    ######################################################################################################
    ######################################################################################################

        robot = Robot(USE_CUDA=True, LATERAL=True, IS_TESTING=TEST)
        
        robot.grasp_lateral(position=[-0.516, -0.05000000000000002, 0.2], best_rotation_angle=0)

        for iter in range(0, 10001):
            print('*'*100)
            print('*'*100)
            print('*'*25, ' Iteration: {} '.format(iter+1), '*'*25)
            print('*'*100)
            print('*'*100)

            robot.get_action(iter)

            if (iter+1) % 5 == 0:
                robot.reset_cup()
                robot.grasp_lateral(position=[-0.516, -0.05000000000000002, 0.2], best_rotation_angle=0)

            if (iter+1) % 50 == 0 and not TEST:
                robot.save_snapshot(iter)

        # Stop simulation
        print('Stopping Simulation...')
        client.simxStopSimulation(client.simxServiceCall())
        print('Simulation Over.')


        

        # NUM_EPISODES = 1000

        # for i_episode in range(1, NUM_EPISODES+1):
            


