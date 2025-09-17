#!/usr/bin/env python

"""teleop_with_recording.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.
"""

import json
import pdb
import time
import argparse
import threading
import os
import numpy as np
import h5py
import logging
from datetime import datetime

# Import utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

# Import Flexiv RDK python libraries
import sys
import flexivrdk
import quaternion
from quest_receive import quest_teleop

# Import Realsense python libraries
from realsense_record import RealSenseModule, get_rgbd
from pdb import set_trace


class TrajectoryRecorder:
    def __init__(self, output_file):
        self.timestamps = []
        self.q_list = []
        self.theta_list = []
        self.dq_list = []
        self.dtheta_list = []
        self.tau_list = []
        self.tau_des_list = []
        self.tau_dot_list = []
        self.tau_ext_list = []
        self.tcp_pose_list = []
        self.tcp_pose_d_list = []
        self.tcp_velocity_list = []
        self.camera_pose_list = []
        self.flange_pose_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []

        self.output_file = output_file

        # Whether recording
        self.is_recording = True

        self.cameras = RealSenseModule()

        self.wrist_image_list = []
        self.image_list = []
        self.wrist_depth_list = []
        self.whole_depth_list = []
        self.action_list = []

    def add_state(self, robot_states, gripper_states, quest_input=None):
        """Add state data for one timestep"""
        if not self.is_recording:
            return

        try:
            self.timestamps.append(time.time())
            self.q_list.append([float(i) for i in robot_states.q])
            self.theta_list.append([float(i) for i in robot_states.theta])
            self.dq_list.append([float(i) for i in robot_states.dq])
            self.dtheta_list.append([float(i) for i in robot_states.dtheta])
            self.tau_list.append([float(i) for i in robot_states.tau])
            self.tau_des_list.append([float(i) for i in robot_states.tauDes])
            self.tau_dot_list.append([float(i) for i in robot_states.tauDot])
            self.tau_ext_list.append([float(i) for i in robot_states.tauExt])
            self.tcp_pose_list.append([float(i) for i in robot_states.tcpPose])
            self.tcp_pose_d_list.append([float(i) for i in robot_states.tcpPoseDes])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcpVel])
            self.camera_pose_list.append([float(i) for i in robot_states.camPose])
            self.flange_pose_list.append([float(i) for i in robot_states.flangePose])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ftSensorRaw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.extWrenchInTcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.extWrenchInBase])

            self.gripper_width_list.append(float(gripper_states.width))

            image, whole_depth, _, wrist_image, wrist_depth, _ = get_rgbd(self.cameras)

            self.wrist_image_list.append(np.array(wrist_image).copy())
            self.image_list.append(np.array(image).copy())

        except KeyboardInterrupt:
            # Don't save current frame, re-raise exception
            raise
        except Exception as e:
            print(f"Error adding state data: {str(e)}")

    def add_action(self, offset_pos, offset_quat, gripper_close):
        """Add action data for one timestep"""
        if not self.is_recording:
            return

        action = [offset_pos[0], offset_pos[1], offset_pos[2], 
                 offset_quat.w, offset_quat.x, offset_quat.y, offset_quat.z, 
                 gripper_close]
        self.action_list.append([float(i) for i in action])

    def align_frames(self):
        """Align all data arrays to the same length"""
        if len(self.action_list) == (len(self.timestamps) - 1):
            self.action_list.append(self.action_list[-1])
        if len(self.image_list) == (len(self.timestamps) - 1):
            self.image_list.append(self.image_list[-1])
        if len(self.wrist_image_list) == (len(self.timestamps) - 1):
            self.wrist_image_list.append(self.wrist_image_list[-1])

        assert len(self.action_list) == len(self.timestamps), \
                f"Action length ({len(self.action_list)}) doesn't match timestamps length ({len(self.timestamps)})"
        assert len(self.image_list) == len(self.timestamps), \
                f"Image length ({len(self.image_list)}) doesn't match timestamps length ({len(self.timestamps)})"
        assert len(self.wrist_image_list) == len(self.timestamps), \
                f"Wrist image length ({len(self.wrist_image_list)}) doesn't match timestamps length ({len(self.timestamps)})"

    def save_trajectory(self, task):
        """Save entire trajectory data to a single HDF5 file with compression"""
        self.align_frames()

        print("Saving file to HDF5 format with compression...")

        with h5py.File(self.output_file, 'w') as hf:
            # Store scalar data with compression
            hf.create_dataset('timestamps', data=np.array(self.timestamps), compression='gzip')
            hf.create_dataset('q', data=np.array(self.q_list), compression='gzip')
            hf.create_dataset('theta', data=np.array(self.theta_list), compression='gzip')
            hf.create_dataset('dq', data=np.array(self.dq_list), compression='gzip')
            hf.create_dataset('dtheta', data=np.array(self.dtheta_list), compression='gzip')
            hf.create_dataset('tau', data=np.array(self.tau_list), compression='gzip')
            hf.create_dataset('tau_des', data=np.array(self.tau_des_list), compression='gzip')
            hf.create_dataset('tau_dot', data=np.array(self.tau_dot_list), compression='gzip')
            hf.create_dataset('tau_ext', data=np.array(self.tau_ext_list), compression='gzip')
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list), compression='gzip')
            hf.create_dataset('tcp_pose_d', data=np.array(self.tcp_pose_d_list), compression='gzip')
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list), compression='gzip')
            hf.create_dataset('camera_pose', data=np.array(self.camera_pose_list), compression='gzip')
            hf.create_dataset('flange_pose', data=np.array(self.flange_pose_list), compression='gzip')
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list), compression='gzip')
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list), compression='gzip')
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list), compression='gzip')
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list), compression='gzip')
            hf.create_dataset('action', data=np.array(self.action_list), compression='gzip')

            # Store image data with compression (using higher compression for images)
            wrist_images = np.array(self.wrist_image_list)
            images = np.array(self.image_list)

            hf.create_dataset('wrist_image', data=wrist_images, 
                             compression='gzip', compression_opts=9, 
                             chunks=(1, wrist_images.shape[1], wrist_images.shape[2], wrist_images.shape[3]))

            hf.create_dataset('image', data=images, 
                             compression='gzip', compression_opts=9,
                             chunks=(1, images.shape[1], images.shape[2], images.shape[3]))

            # Store metadata
            hf.attrs['instruction'] = task
            hf.attrs['num_frames'] = len(self.timestamps)
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hf.attrs['image_shape'] = str(images.shape[1:])
            hf.attrs['wrist_image_shape'] = str(wrist_images.shape[1:])

        print(f"Task name: {task}")
        print(f"Epoch sequence length: {len(self.timestamps)}")
        print(f"File saved successfully to: {self.output_file}")

def print_description():
    """
    Print tutorial description.
    """
    print("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")
    print()

def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose"""
    robot_states = robot.states()

    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array([current_tcp_pose[0], current_tcp_pose[1], current_tcp_pose[2]])
    current_tcp_quat = quaternion.quaternion(current_tcp_pose[3], current_tcp_pose[4], 
                                           current_tcp_pose[5], current_tcp_pose[6])

    gripper_states = gripper.states()

    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states

def main(task, path, frequency):
    """Main function for teleoperation with recording"""
    # Define alias
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    mode = flexivrdk.Mode

    # Print description
    logging.info("Script description:")
    print_description()

    # Create output directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.h5"
    output_file = os.path.join(path, filename)

    # Create trajectory recorder with HDF5 format
    recorder = TrajectoryRecorder(output_file)

    # Initialize Quest teleoperation controller
    quest_controller = quest_teleop()

    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot("Rizon 4s-063034")

        # Clear fault on robot server if any
        if robot.fault():
            logging.warning("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.ClearFault()
            time.sleep(2)
            # Check again
            if robot.fault():
                logging.error("Fault cannot be cleared, exiting ...")
                return
            logging.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        logging.info("Enabling robot ...")
        robot.Enable()

        # Wait for the robot to become operational
        seconds_waited = 0
        while not robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                logging.warninging(
                    "Still waiting for robot to become operational, please check that the robot 1) "
                    "has no fault, 2) is in [Auto (remote)] mode")

        logging.info("Robot is now operational")
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)

        # Instantiate gripper control interface
        gripper = flexivrdk.Gripper(robot)
        gripper.Enable("Flexiv-GN01")

        logging.info("Opening gripper")
        gripper.Move(0.09, 0.1, 20)
        time.sleep(1)

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        logging.info("Executing primitive: Home")

        # # Send command to robot
        # robot.ExecutePrimitive("Home", dict())

        # while not robot.primitive_states()["reachedTarget"]:
        #     # Print current primitive states
        #     print(robot.primitive_states())
        #     time.sleep(1)

        # Switch to cartesian motion force mode for teleoperation
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        logging.info("Starting teleoperation with trajectory recording...")
        logging.info(f"Recording will be saved to: {output_file}")

        last_input = None
        frame_cnt = 0
        last_robot_states, last_tcp_pos, last_tcp_quat, last_gripper_states = get_cur_pose(robot, gripper)

        try:
            while True:
                # Get current robot states
                robot_states = robot.states()
                gripper_states = gripper.states()

                # Update Quest controller's joint states
                joint_states = robot_states.q
                quest_controller.joint_states = np.array(joint_states)

                # Get Quest controller input
                current_input, _, _ = quest_controller.get_input_frame()

                if current_input is None:
                    time.sleep(0.02)
                    continue

                if current_input.get('Y', False) and current_input.get('B', False):
                    logging.info("Y + B detected, recording stopped by user. Saving trajectory...")
                    break

                if last_input is None:
                    last_input = current_input

                # Record robot state
                recorder.add_state(robot_states, gripper_states)

                # Get current robot TCP pose
                current_tcp_pose = robot_states.tcpPose
                current_tcp_pos = np.array([current_tcp_pose[0], current_tcp_pose[1], current_tcp_pose[2]])
                current_tcp_quat = quaternion.quaternion(current_tcp_pose[3], current_tcp_pose[4], 
                                                       current_tcp_pose[5], current_tcp_pose[6])

                # Get Quest controller input pose
                current_input_pos = np.array([current_input['rightPos']["z"], 
                                            -current_input['rightPos']["x"], 
                                            current_input['rightPos']["y"]])
                current_input_quat = quaternion.quaternion(current_input['rightRot']["w"], 
                                                         current_input['rightRot']["z"], 
                                                         current_input['rightRot']["x"], 
                                                         current_input['rightRot']["y"])

                # Perform teleoperation when Quest right controller button is pressed
                if current_input.get('rightHand', 0) > 0.5:
                    if last_input.get('rightHand', 0) <= 0.5:
                        # Record starting pose
                        start_tcp_pos = current_tcp_pos
                        start_tcp_quat = current_tcp_quat
                        start_input_pos = current_input_pos
                        start_input_quat = current_input_quat

                    # Calculate position and rotation offset
                    offset_pos = current_input_pos - start_input_pos
                    offset_quat = quaternion.quaternion.inverse(start_input_quat) * current_input_quat

                    # Calculate target pose
                    pos = start_tcp_pos + offset_pos
                    quat = start_tcp_quat * offset_quat

                    # Send target pose to robot
                    target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    new_target = [pos[0], pos[1], pos[2], quat.w, quat.x, quat.y, quat.z]
                    robot.sendCartesianMotionForce(new_target, target_wrench)

                    # Control gripper
                    gripper_close = 0.09 * (1 - current_input.get('rightIndex', 0)) + 0.01
                    gripper.Move(gripper_close, 0.1, 20)

                    recorder.add_action(pos, quat, gripper_close)
                else:
                    # Record current pose as action when not teleoperating
                    recorder.add_action(current_tcp_pos, current_tcp_quat, gripper_states.width)

                # Update last input
                last_input = current_input
                frame_cnt += 1
                if frame_cnt % frequency == 0:
                    print(f"Recorded {frame_cnt} frames...")

        except KeyboardInterrupt:
            logging.info("Teleoperation interrupted by user. Saving trajectory...")

    except Exception as e:
        # Print exception error message
        logging.error(str(e))
        import traceback
        traceback.print_exc()
    finally:
        # Send command to robot
        robot.ExecutePrimitive("Home", dict())
        logging.info("Executing primitive: Move to record HOME Joints")

        while not robot.primitive_states()["reachedTarget"]:
            # Print current primitive states
            print(robot.primitive_states())
            time.sleep(1)

        # Record final robot state
        cur_robot_states, cur_tcp_pos, cur_tcp_quat, cur_gripper_states = get_cur_pose(robot, gripper)
        recorder.add_state(cur_robot_states, cur_gripper_states)
        recorder.add_action(cur_tcp_pos, cur_tcp_quat, cur_gripper_states.width)

        # Ensure trajectory is saved
        recorder.save_trajectory(task)
        logging.info(f"Trajectory saved to {recorder.output_file}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording")
    parser.add_argument("--path", type=str, default="./teleop_recordings/", help="Path to save recordings")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")

    args = parser.parse_args()

    main(task=args.task, path=args.path, frequency=args.frequency)