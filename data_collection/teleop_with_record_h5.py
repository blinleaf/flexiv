#!/usr/bin/env python

"""teleop_with_recording.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.
"""

import json
import time
import argparse
import threading
import os
import numpy as np
import h5py
import spdlog
from datetime import datetime

# Import utility methods
from utility import list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion
from quest_receive import QuestTeleop

# Import Realsense python libraries
from realsense_record import RealSenseModule, get_rgbd, CameraConfig


class TrajectoryRecorder:
    def __init__(self, output_file, camera_config=None):
        self.timestamps = []
        self.tcp_pose_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        self.camera_images_list = {}
        self.action_list = []

        self.output_file = output_file
        self.is_recording = True
        self.logger = spdlog.ConsoleLogger("Recorder")
        # Initialize RealSenseModule
        self.cameras = RealSenseModule(camera_config)
        # Initialize lists for each camera
        self.num_cameras = len(self.cameras.serial_numbers)
        self.logger.info(f"Initialized {self.num_cameras} cameras")
        for i in range(self.num_cameras):
            self.camera_images_list[f'cam{i+1}'] = []

    def add_state(self, robot_states, gripper_states):
        """Add state data for one timestep"""
        if not self.is_recording:
            return

        try:
            self.timestamps.append(time.time())
            self.tcp_pose_list.append([float(i) for i in robot_states.tcp_pose])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcp_vel])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ft_sensor_raw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.ext_wrench_in_tcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.ext_wrench_in_world])
            self.gripper_width_list.append(float(gripper_states.width))

            camera_data = get_rgbd(self.cameras)
            if len(camera_data) != self.num_cameras:
                self.logger.error(f"Expected {self.num_cameras} camera feeds, but got {len(camera_data)}")
                return
            for i, (image, _, _) in enumerate(camera_data):
                cam_key = f'cam{i+1}'
                if cam_key not in self.camera_images_list:
                    self.logger.warn(f"Camera key {cam_key} not initialized, creating now")
                    self.camera_images_list[cam_key] = []
                self.camera_images_list[cam_key].append(np.array(image, copy=True))

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")

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
        min_length = min(len(self.timestamps), 
                        len(self.action_list) + 1, 
                        *[len(self.camera_images_list[f'cam{i+1}']) + 1 for i in range(self.num_cameras)])
        
        if len(self.action_list) < min_length:
            self.action_list.append(self.action_list[-1])
        
        for cam_name in self.camera_images_list:
            if len(self.camera_images_list[cam_name]) < min_length:
                self.camera_images_list[cam_name].append(self.camera_images_list[cam_name][-1])

        for attr in ['timestamps', 'action_list']:
            setattr(self, attr, getattr(self, attr)[:min_length])

        for cam_name in self.camera_images_list:
            self.camera_images_list[cam_name] = self.camera_images_list[cam_name][:min_length]

        assert len(self.action_list) == len(self.timestamps), \
            f"Action length ({len(self.action_list)}) doesn't match timestamps ({len(self.timestamps)})"
        for i in range(self.num_cameras):
            cam_name = f'cam{i+1}'
            assert len(self.camera_images_list[cam_name]) == len(self.timestamps), \
                f"{cam_name} image length ({len(self.camera_images_list[cam_name])}) doesn't match timestamps ({len(self.timestamps)})"

    def save_trajectory(self, task):
        """Save trajectory data to HDF5 file, images as uint8, other data as float32"""
        self.align_frames()
        self.logger.info(f"Saving trajectory to {self.output_file}...")
        
        def save_images(hf, images, cam_name):
            # Convert images to uint8 and save with compression
            images = images.astype(np.uint8)
            hf.create_dataset(cam_name, data=images, 
                            chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                            dtype='uint8')
            hf.attrs[f'{cam_name}_shape'] = str(images.shape[1:])
        
        with h5py.File(self.output_file, 'w', libver='latest', rdcc_nbytes=1024*1024*100) as hf:
            # Save non-image data as float32
            hf.create_dataset('timestamps', data=np.array(self.timestamps, dtype=np.float32))
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list, dtype=np.float32))
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list, dtype=np.float32))
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list, dtype=np.float32))
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list, dtype=np.float32))
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list, dtype=np.float32))
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list, dtype=np.float32))
            hf.create_dataset('action', data=np.array(self.action_list, dtype=np.float32))
            
            # Save metadata
            hf.attrs['instruction'] = task
            hf.attrs['num_frames'] = len(self.timestamps)
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hf.attrs['num_cameras'] = self.num_cameras
            
            # Save images asynchronously
            threads = []
            for i in range(self.num_cameras):
                cam_name = f'cam{i+1}'
                images = np.array(self.camera_images_list[cam_name])
                thread = threading.Thread(target=save_images, args=(hf, images, cam_name))
                threads.append(thread)
                thread.start()
            
            # Wait for all image saving to complete
            for thread in threads:
                thread.join()
        
        self.logger.info(f"Task: {task}, Frames: {len(self.timestamps)}, Saved to: {self.output_file}")

def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose"""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    gripper_states = gripper.states()
    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states

def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30):
    """Main function for teleoperation with recording"""
    logger = spdlog.ConsoleLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(path, f"trajectory_{timestamp}.h5")
    
    # Initialize CameraConfig
    camera_config = CameraConfig(
        real_time_view=True,
        rgb_size=(rgb_width, rgb_height),
        depth_size=(rgb_width, rgb_height),
        fps=fps,
        save_freq=frequency
    )
    
    recorder = TrajectoryRecorder(output_file, camera_config)
    quest_controller = QuestTeleop()

    try:
        # RDK Initialization
        robot = flexivrdk.Robot("Rizon 4s-063036")
        if robot.fault():
            logger.warn("Fault on robot server, trying to clear...")
            robot.ClearFault()
            time.sleep(2)
            if robot.fault():
                logger.error("Fault cannot be cleared, exiting...")
                return
            logger.info("Fault cleared")

        logger.info("Enabling robot...")
        robot.Enable()
        seconds_waited = 0
        while not robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                logger.warn("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                return
        logger.info("Robot operational")
        
        gripper = flexivrdk.Gripper(robot)
        gripper.Enable("GripperDahuanModbus")
        
        logger.info("Opening gripper")
        gripper.Move(0.1, 0.2, 20)
        while robot.busy():
            time.sleep(1)

        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        while robot.busy():
            time.sleep(1)

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )
        while not robot.primitive_states()["terminated"]:
            time.sleep(1)
        logger.info("Sensor zeroing complete")

        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info(f"Starting teleoperation, recording to: {output_file}")

        last_input = None
        frame_cnt = 0
        last_robot_states, last_tcp_pos, last_tcp_quat, last_gripper_states = get_cur_pose(robot, gripper)

        while True:
            robot_states = robot.states()
            gripper_states = gripper.states()
            quest_controller.joint_states = np.array(robot_states.q)
            current_input, _, _ = quest_controller.get_input_frame()

            if current_input is None:
                logger.warn("No input from Quest controller, waiting...")
                time.sleep(0.02)
                continue

            if current_input.get('Y', False) and current_input.get('B', False):
                logger.info("Y + B detected, stopping recording...")
                break

            if last_input is None:
                last_input = current_input

            recorder.add_state(robot_states, gripper_states)
            current_tcp_pose = robot_states.tcp_pose
            current_tcp_pos = np.array(current_tcp_pose[:3])
            current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
            
            current_input_pos = np.array([current_input['rightPos']["z"], 
                                       -current_input['rightPos']["x"], 
                                       current_input['rightPos']["y"]])
            current_input_quat = quaternion.quaternion(current_input['rightRot']["w"], 
                                                    current_input['rightRot']["z"], 
                                                    current_input['rightRot']["x"], 
                                                    current_input['rightRot']["y"])

            if current_input.get('rightHand', 0) > 0.5:
                if last_input.get('rightHand', 0) <= 0.5:
                    start_tcp_pos = current_tcp_pos
                    start_tcp_quat = current_tcp_quat
                    start_input_pos = current_input_pos
                    start_input_quat = current_input_quat

                offset_pos = current_input_pos - start_input_pos
                offset_quat = quaternion.quaternion.inverse(start_input_quat) * current_input_quat
                pos = start_tcp_pos + offset_pos
                quat = start_tcp_quat * offset_quat
                robot.SendCartesianMotionForce([*pos, quat.w, quat.x, quat.y, quat.z], [0.0] * 6)
                gripper_close = 0.09 * (1 - current_input.get('rightIndex', 0)) + 0.01
                gripper.Move(gripper_close, 0.2, 20)
                recorder.add_action(pos, quat, gripper_close)
            else:
                recorder.add_action(current_tcp_pos, current_tcp_quat, gripper_states.width)

            last_input = current_input
            frame_cnt += 1
            if frame_cnt % frequency == 0:
                logger.info(f"Recorded {frame_cnt} frames...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user, saving trajectory...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.align_frames()
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        while robot.busy():
            time.sleep(0.02)

        recorder.cameras.cleanup()
        recorder.save_trajectory(task)
        logger.info(f"Trajectory saved to {recorder.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording")

    current_date = datetime.now().strftime("%Y-%m-%d")
    default_path = f"../data/flexiv/teleop_recordings/{current_date}/"
    parser.add_argument("--path", type=str, default=default_path, help="Path to save HDF5 files")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=480, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    main(task=args.task, path=args.path, frequency=args.frequency, 
         rgb_width=args.rgb_width, rgb_height=args.rgb_height, fps=args.fps)