#!/usr/bin/env python3


# Make sure to have CoppeliaSim running
#
# Do not launch simulation, and make sure that the B0 resolver
# is running. Then run "simpleTest"
#
# The client side (i.e. "simpleTest") depends on:
#
# b0RemoteApi (Python script), which depends on:
# msgpack (Python messagePack binding, install with "pip install msgpack")
# b0.py (Python script), which depends on:
# b0 (shared library), which depends on:
# boost_chrono (shared library)
# boost_system (shared library)
# boost_thread (shared library)
# libzmq (shared library)

# Built-in lib imports
import time, random

# Standard Library imports
import numpy as np

# Third-Party imports
import b0RemoteApi




with b0RemoteApi.RemoteApiClient('b0RemoteApi_V-REP','b0RemoteApi') as client:

    # Make sure simulation is not running
    client.simxStopSimulation(client.simxDefaultPublisher())

    # Global variables
    cube_position = 0
    doNextStep = True
    target_right_pose = []


    # Callbacks
    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        # print('Simulation step done. Simulation time: ',simTime);
        global doNextStep
        doNextStep=True

    def target_right_cb(msg):
        # print('Received target_right pose.', msg[0])
        # print(msg[1])
        global target_right_pose 
        target_right_pose = msg[1]

    def cube_cb(msg):
        global cube_position
        cube_position = msg[1]




    class Robot():
        def __init__(self):

            # Create object handles
            _, self.target_right_handle = client.simxGetObjectHandle("target_right", client.simxServiceCall())
            _, self.connector_handle = client.simxGetObjectHandle('RG2_attachPoint', client.simxServiceCall())
            _, self.sensor_handle = client.simxGetObjectHandle('RG2_attachProxSensor', client.simxServiceCall())
            _, self.gripper_joint_handle = client.simxGetObjectHandle('RG2_openCloseJoint#0', client.simxServiceCall())
            _, self.cube_handle = client.simxGetObjectHandle("cube", client.simxServiceCall())
            _, self.right_force_sensor_handle = client.simxGetObjectHandle("RG2_rightForceSensor#0", client.simxServiceCall())
            _, self.vision_sensor_handle = client.simxGetObjectHandle('vision_sensor', client.simxServiceCall())
            

            # Subscribers
            self.reference_frame = -1
            client.simxGetObjectPosition(self.target_right_handle,self.reference_frame, client.simxDefaultSubscriber(target_right_cb))
            client.simxGetObjectPosition(self.cube_handle, self.reference_frame, client.simxDefaultSubscriber(cube_cb))
            # client.simxReadVisionSensor(self.vision_sensor_handle, client.simxDefaultSubscriber(vision_sensor_image_and_depth_cb))
            
            # Subscribers and Callbacks
            client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))

            # Set synchornous 
            client.simxSynchronous(True)

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

        def start(self):
            # Start simulation and log message
            client.simxAddStatusbarMessage("Starting!!!", client.simxDefaultPublisher())
            client.simxStartSimulation(client.simxDefaultPublisher())       

            # Set initial poses
            sig_1 = client.simxSetObjectOrientation(self.target_right_handle, self.reference_frame, [-1.57, 0, -1.3], client.simxServiceCall())
            assert sig_1[0] == True

            # Get initial poses
            _, gripper_position = client.simxGetJointPosition(self.gripper_joint_handle, client.simxServiceCall())
            print("Gripper Initial Position: {}".format(gripper_position))

        def mse(self, target, current):
            n = len(target)
            error = 0
            for i in range(3):
                error += (target[i] - current[i])**2
            error /= n

            return error


        def set_pose(self, pose, TYPE):
            if TYPE == "position" :
                current_position = client.simxGetObjectPosition(self.target_right_handle, self.reference_frame, client.simxServiceCall())
                assert current_position[0] == True
                
                n_steps = 10
                diff_vector = []
                for i in range(len(pose)):
                    diff_vector.append(pose[i] - current_position[1][i])
                step = [x/n_steps for x in diff_vector]

                tolerance = 0.0001
                while self.mse(current_position[1], pose) > tolerance:
                    print('Error: ', self.mse(current_position[1], pose))
                    target_position = [current_position[1][i]+step[i] for i in range(len(pose))]
                    response = client.simxSetObjectPosition(self.target_right_handle, self.reference_frame, target_position,
                                                            client.simxServiceCall())
                    assert response[0] == True

                    current_position = client.simxGetObjectPosition(self.target_right_handle, self.reference_frame, client.simxServiceCall())

                    client.simxSynchronousTrigger()
                    client.simxSpinOnce()
            print("Current Position: {}".format(current_position[1]))
            print("Target Position: {}".format(pose))


    # Main

    robot = Robot()
    robot.start()

    flag = True
    start_time = time.time()
    
    while time.time() - start_time < 6:

        if doNextStep:
            doNextStep = False
            if time.time() - start_time > 2 and flag == True:
                print('yes')
                flag = False

                robot.open_gripper()
        
                cube_position = [x for x in cube_position]
                cube_position[2] += 0.01 # 0.015
                robot.set_pose(cube_position, "position")
                
                robot.close_gripper()

                cube_position = [x/2 for x in cube_position]
                cube_position[2] += 0.5
                robot.set_pose(cube_position, "position")



            client.simxSynchronousTrigger()

        client.simxSpinOnce()
    # Stop simulation
    client.simxStopSimulation(client.simxDefaultPublisher())
