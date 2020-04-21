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

# Standard lib imports
import time

# Third-Party imports
import ..b0RemoteApi


with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApiAddOn') as client:
    doNextStep=True

    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step done. Simulation time: ',simTime);
        global doNextStep
        doNextStep=True

    def vision_sensor_cb(msg):
        print('Received image.',msg[1])
        client.simxSetVisionSensorImage(side_vision_sensor_handle[1], False, msg[2], client.simxDefaultPublisher())

    # Publish to status bar
    client.simxAddStatusbarMessage("Hello Wolrd!!", client.simxDefaultPublisher())

    # Create object handles
    vision_sensor_handle = client.simxGetObjectHandle("vision_sensor", client.simxServiceCall())
    side_vision_sensor_handle = client.simxGetObjectHandle("side_vision_sensor", client.simxServiceCall())


    # Create a dedicated sub
    #dedicatedSub=client.simxCreateSubscriber(imageCallback,1,True)
    #client.simxGetVisionSensorImage(visionSensorHandle[1],False,dedicatedSub)

    # Create a normal sub for vision sensor
    client.simxGetVisionSensorImage(vision_sensor_handle[1], False, client.simxDefaultSubscriber(vision_sensor_cb))

    # Callback for simulation step done
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))

    # Set synchornous 
    client.simxSynchronous(True)

    # Start simulation
    client.simxStartSimulation(client.simxDefaultPublisher())

    # Main

    # Assert handles retrieval 
    assert (vision_sensor_handle[0] and side_vision_sensor_handle[0]) == 1

    start_time = time.time()

    while time.time() - start_time < 3:
        if doNextStep:
            doNextStep = False
            client.simxSynchronousTrigger()
        client.simxSpinOnce()
    # Stop simulation
    client.simxStopSimulation(client.simxDefaultPublisher())

