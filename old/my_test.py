#!/usr/bin/env python3

# Built-in lib imports
import time, random, struct, pickle

# Standard Library imports
import numpy as np
import math

# Third-Party imports
import b0RemoteApi

import skimage.transform as trans
from utils_warp import convert_image_np, normalize_transforms, rotatepoints, show_image, prepare_img




with b0RemoteApi.RemoteApiClient('b0RemoteApi_V-REP','b0RemoteApi', timeout=5) as client:

    # Make sure simulation is not running
    client.simxStopSimulation(client.simxDefaultPublisher())

    # Global variables
    doNextStep = True
    rgb_vision_msg = None
    d_vision_msg = None

    # Callbacks
    def simulationStepStarted(msg):
        # simTime=msg[1][b'simulationTime'];
        # print('Simulation step started. Simulation time: ',simTime)
        pass

    def simulationStepDone(msg):
        global doNextStep
        doNextStep = True

    def rgb_vision_cb(msg):
        global rgb_vision_msg
        rgb_vision_msg = msg[2]
        # _, side_vision_sensor_handle = client.simxGetObjectHandle('side_vision_sensor', client.simxServiceCall())
        # client.simxSetVisionSensorImage(side_vision_sensor_handle, False, rgb_vision_msg, client.simxDefaultPublisher())

    def d_vision_cb(msg):
        global d_vision_msg
        d_vision_msg = msg[2]


    def vision_sensor_cb(msg):
        start_time = time.time()
        client.simxGetVisionSensorDepthBuffer(vision_sensor_handle, True, False, client.simxServiceCall())
        end_time = time.time()
        print(' Elapsed: {}'.format(end_time-start_time))
        client.simxSetVisionSensorImage(side_vision_sensor_handle, False, rgb_vision_msg, client.simxDefaultPublisher())

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
        print(color_img.shape)

        return color_img

    def get_camera_data():

        start_time = time.time()
        rgb_ret, rgb_res, rgb_img_raw = client.simxGetVisionSensorImage(vision_sensor_handle, False, client.simxServiceCall())
        end_time = time.time()
        # print('Elapsed rgb capture: {}'.format(end_time-start_time))

        start_time = time.time()
        d_ret, d_res, d_img_raw = client.simxGetVisionSensorDepthBuffer(vision_sensor_handle, True, True, client.simxServiceCall())
        end_time = time.time()
        # print('Elapsed depth capture: {}'.format(end_time-start_time))

        assert (d_ret and rgb_ret) == True

        client.simxSetVisionSensorImage(side_vision_sensor_handle, False, rgb_img_raw, client.simxDefaultPublisher())
        
        depth_img = depth_img_from_bytes(d_img_raw, rgb_res)
        # np.save('depth_img', depth_img)

        color_img = rgb_img_from_bytes(rgb_img_raw, rgb_res)
        # np.save('color_img', color_img)

        return rgb_img_raw, d_img_raw


def prepare_network_input(rgb_img_raw, d_img_raw):
        # New stuff start heres
        color_img = prepare_img(color_img, mode='rgb')
        depth_img = prepare_img(depth_img, mode='depth')
        cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        workspace_limits = np.asarray([[-2, 2], [-6, -2], [-5, -4]]) # note workspace limits
        heightmap_resolution = 0.02 # No idea
        # cam pose in vrep
        cam_pose = np.asarray([[ 1, 0, 0, 0], [ 0, -0.70710679, -0.70710678, 1], [ 0, 0.70710678, -0.70710679, 0.5]])

        color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution)
        depth_heightmap[np.isnan(depth_heightmap)] = 0

        input_color_data, input_depth_data = get_input_tensors(color_heightmap, depth_heightmap)

        # Here comes the rotation step, which we will SKIP FOR NOW


    # Open Gripper
    def open_gripper(self):
        motor_velocity = 0.5 # m/s
        motor_force = 100 # N
        client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
        client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
        gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
        while gripper_position[1] < 0:

            gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())

            client.simxSynchronousTrigger()
            client.simxSpinOnce()

    # Close Gripper
    def close_gripper(self):
        motor_velocity = -0.5 # m/s
        motor_force = 100 # N
        client.simxSetJointForce(self.gripper_joint_handle, motor_force, client.simxServiceCall())
        client.simxSetJointTargetVelocity(self.gripper_joint_handle, motor_velocity, client.simxServiceCall())
        gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
        right_force_sensor_feedback= client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
        while gripper_position[1] > -0.047 and right_force_sensor_feedback[2][2] > -90:
            gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
            right_force_sensor_feedback = client.simxReadForceSensor(self.right_force_sensor_handle, client.simxServiceCall())
            print("right_force_sensor_feedback: ", right_force_sensor_feedback[2])

            client.simxSynchronousTrigger()
            client.simxSpinOnce()



    ###################################################

    # Create object handles
    _, target_right_handle = client.simxGetObjectHandle("target_right", client.simxServiceCall())
    _, connector_handle = client.simxGetObjectHandle('RG2_attachPoint', client.simxServiceCall())
    _, sensor_handle = client.simxGetObjectHandle('RG2_attachProxSensor', client.simxServiceCall())
    _, gripper_joint_handle = client.simxGetObjectHandle('RG2_openCloseJoint#0', client.simxServiceCall())
    _, cube_handle = client.simxGetObjectHandle("cube", client.simxServiceCall())
    _, right_force_sensor_handle = client.simxGetObjectHandle("RG2_rightForceSensor#0", client.simxServiceCall())
    _, vision_sensor_handle = client.simxGetObjectHandle('vision_sensor', client.simxServiceCall())
    _, side_vision_sensor_handle = client.simxGetObjectHandle('side_vision_sensor', client.simxServiceCall())


    ## Create subscribers
    # client.simxGetVisionSensorImage(vision_sensor_handle, False, client.simxDefaultSubscriber(rgb_vision_cb))
    # client.simxReadVisionSensor(vision_sensor_handle, client.simxDefaultSubscriber(vision_sensor_cb))

    # Initiate simulation options
    client.simxSynchronous(True)
    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))
    client.simxStartSimulation(client.simxDefaultPublisher())


    client.simxAddStatusbarMessage("Starting!!!", client.simxDefaultPublisher())
    print('Started Simulation!')

    camera_position = client.simxGetObjectPosition(vision_sensor_handle,  -1, client.simxServiceCall())
    camera_orientation = client.simxGetObjectOrientation(vision_sensor_handle,  -1, client.simxServiceCall())
    rotation_matrix = euler2rotm(camera_orientation[1])
    camera_transformation = client.simxGetObjectMatrix(vision_sensor_handle,  -1, client.simxServiceCall())

    print(camera_position[1])
    print(rotation_matrix)
    print(camera_transformation[1])


    startTime=time.time()
    while time.time()<startTime+5: 
        if doNextStep:
            doNextStep=False
            ###
            get_camera_data()
            ###
            client.simxSynchronousTrigger()
        client.simxSpinOnce()
    client.simxStopSimulation(client.simxDefaultPublisher())

    # Stop simulation
    print('Stopping Simulation...')
    client.simxStopSimulation(client.simxDefaultPublisher())
    print('Simulation Over.')
