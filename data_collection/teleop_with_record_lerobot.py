#!/usr/bin/env python

"""teleop_with_lerobot_video.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording using LeRobotDataset, storing camera data as videos.
"""

import argparse
import logging
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Import utility methods
from utility import quat2eulerZYX, list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion

# Import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import Quest and RealSense modules
from quest_receive import QuestTeleop
from realsense_record import RealSenseModule, get_rgbd, CameraConfig


def get_features(num_cameras):
    """Define the features for the LeRobotDataset, with camera data as videos."""
    features = {
        "observation.state.q": {"dtype": "float32", "shape": [7], "names": [f"q{i}" for i in range(7)]},
        "observation.state.theta": {"dtype": "float32", "shape": [7], "names": [f"theta{i}" for i in range(7)]},
        "observation.state.dq": {"dtype": "float32", "shape": [7], "names": [f"dq{i}" for i in range(7)]},
        "observation.state.dtheta": {"dtype": "float32", "shape": [7], "names": [f"dtheta{i}" for i in range(7)]},
        "observation.state.tau": {"dtype": "float32", "shape": [7], "names": [f"tau{i}" for i in range(7)]},
        "observation.state.tau_des": {"dtype": "float32", "shape": [7], "names": [f"tau_des{i}" for i in range(7)]},
        "observation.state.tau_dot": {"dtype": "float32", "shape": [7], "names": [f"tau_dot{i}" for i in range(7)]},
        "observation.state.tau_ext": {"dtype": "float32", "shape": [7], "names": [f"tau_ext{i}" for i in range(7)]},
        "observation.state.tcp_pose": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        "observation.state.tcp_velocity": {"dtype": "float32", "shape": [6], "names": ["vx", "vy", "vz", "wx", "wy", "wz"]},
        "observation.state.flange_pose": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        "observation.state.ft_sensor_raw": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
        "observation.state.f_ext_tcp_frame": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
        "observation.state.f_ext_base_frame": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
        "observation.state.gripper_width": {"dtype": "float32", "shape": []},
        "action": {"dtype": "float32", "shape": [8], "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_close"]},
    }
    # Add camera features as videos
    for i in range(num_cameras):
        features[f"observation.video.cam{i+1}"] = {"dtype": "video", "shape": [480, 640, 3]}
    return features


def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose."""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    gripper_states = gripper.states()
    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states


def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, save_path="./data", batch_encoding_size=1):
    """Main function for teleoperation with recording using LeRobotDataset, storing camera data as videos."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording using LeRobotDataset, storing camera data as videos.")

    # Initialize CameraConfig
    camera_config = CameraConfig(
        real_time_view=True,
        rgb_size=(rgb_width, rgb_height),
        depth_size=(rgb_width, rgb_height),
        fps=fps,
        save_path=save_path
    )
    cameras = RealSenseModule(camera_config)
    num_cameras = len(cameras.serial_numbers)
    logger.info(f"Initialized {num_cameras} cameras")

    # Define dataset features
    features = get_features(num_cameras)

    # Initialize LeRobotDataset with video storage
    repo_id = f"teleop_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=path,
        robot_type="Rizon 4s",
        use_videos=True,  # Explicitly set to store videos
        batch_encoding_size=batch_encoding_size,
        image_writer_processes=4,  # Adjust based on system capabilities
        image_writer_threads=4
    )
    logger.info(f"Created LeRobotDataset with repo_id: {repo_id}")

    # Initialize Quest controller
    quest_controller = QuestTeleop()

    try:
        # RDK Initialization
        robot = flexivrdk.Robot("Rizon 4s-063034")
        if robot.fault():
            logger.warning("Fault on robot server, trying to clear...")
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
                logger.warning("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                return
        logger.info("Robot operational")

        gripper = flexivrdk.Gripper(robot)
        gripper.Enable("Flexiv-GN01")

        logger.info("Opening gripper")
        gripper.Move(0.1, 0.2, 20)
        while robot.busy():
            time.sleep(1)

        robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(1)

        robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warning("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        while not robot.primitive_states()["terminated"]:
            time.sleep(1)
        logger.info("Sensor zeroing complete")

        robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info(f"Starting teleoperation, recording to: {path}/{repo_id}")

        last_input = None
        frame_cnt = 0
        last_robot_states, last_tcp_pos, last_tcp_quat, last_gripper_states = get_cur_pose(robot, gripper)

        while True:
            robot_states = robot.states()
            gripper_states = gripper.states()
            quest_controller.joint_states = np.array(robot_states.q)
            current_input, _, _ = quest_controller.get_input_frame()

            if current_input is None:
                logger.warning("No input from Quest controller, waiting...")
                time.sleep(0.02)
                continue

            if current_input.get('Y', False) and current_input.get('B', False):
                logger.info("Y + B detected, stopping recording...")
                break

            if last_input is None:
                last_input = current_input

            # Prepare frame data
            frame = {
                "timestamp": time.time(),
                "task": task,
                "observation.state.q": np.array(robot_states.q, dtype=np.float32),
                "observation.state.theta": np.array(robot_states.theta, dtype=np.float32),
                "observation.state.dq": np.array(robot_states.dq, dtype=np.float32),
                "observation.state.dtheta": np.array(robot_states.dtheta, dtype=np.float32),
                "observation.state.tau": np.array(robot_states.tau, dtype=np.float32),
                "observation.state.tau_des": np.array(robot_states.tau_des, dtype=np.float32),
                "observation.state.tau_dot": np.array(robot_states.tau_dot, dtype=np.float32),
                "observation.state.tau_ext": np.array(robot_states.tau_ext, dtype=np.float32),
                "observation.state.tcp_pose": np.array(robot_states.tcp_pose, dtype=np.float32),
                "observation.state.tcp_velocity": np.array(robot_states.tcp_vel, dtype=np.float32),
                "observation.state.flange_pose": np.array(robot_states.flange_pose, dtype=np.float32),
                "observation.state.ft_sensor_raw": np.array(robot_states.ft_sensor_raw, dtype=np.float32),
                "observation.state.f_ext_tcp_frame": np.array(robot_states.ext_wrench_in_tcp, dtype=np.float32),
                "observation.state.f_ext_base_frame": np.array(robot_states.ext_wrench_in_world, dtype=np.float32),
                "observation.state.gripper_width": np.float32(gripper_states.width),
            }

            # Add camera videos
            camera_data = get_rgbd(cameras)
            if len(camera_data) != num_cameras:
                logger.error(f"Expected {num_cameras} camera feeds, but got {len(camera_data)}")
                continue
            for i, (image, _, _) in enumerate(camera_data):
                frame[f"observation.video.cam{i+1}"] = np.array(image, dtype=np.uint8)

            # Compute action
            current_tcp_pose = robot_states.tcp_pose
            current_tcp_pos = np.array(current_tcp_pose[:3])
            current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])

            current_input_pos = np.array([
                current_input['rightPos']["z"],
                -current_input['rightPos']["x"],
                current_input['rightPos']["y"]
            ])
            current_input_quat = quaternion.quaternion(
                current_input['rightRot']["w"],
                current_input['rightRot']["z"],
                current_input['rightRot']["x"],
                current_input['rightRot']["y"]
            )

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
                frame["action"] = np.array([*pos, quat.w, quat.x, quat.y, quat.z, gripper_close], dtype=np.float32)
            else:
                frame["action"] = np.array([*current_tcp_pos, current_tcp_quat.w, current_tcp_quat.x,
                                          current_tcp_quat.y, current_tcp_quat.z, gripper_states.width], dtype=np.float32)

            # Add frame to dataset
            dataset.add_frame(frame)

            last_input = current_input
            frame_cnt += 1
            if frame_cnt % frequency == 0:
                logger.info(f"Recorded {frame_cnt} frames...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user, saving episode...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the current episode
        if dataset.episode_buffer["size"] > 0:
            dataset.save_episode()
            logger.info(f"Episode saved to {path}/{repo_id}")

        # Finalize robot operations
        robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(0.02)

        # Clean up
        cameras.cleanup()
        dataset.stop_image_writer()
        logger.info(f"Dataset saved to {path}/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording using LeRobotDataset, storing camera data as videos")
    parser.add_argument("--path", type=str, default="./teleop_recordings/", help="Path to save recordings")
    parser.add_argument("--frequency", type=int, default=30, help="Logging frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=480, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--save_path", type=str, default="./data", help="Path to save RGB-D data")
    parser.add_argument("--batch_encoding_size", type=int, default=1, help="Number of episodes to batch encode videos")
    args = parser.parse_args()

    main(
        task=args.task,
        path=args.path,
        frequency=args.frequency,
        rgb_width=args.rgb_width,
        rgb_height=args.rgb_height,
        fps=args.fps,
        save_path=args.save_path,
        batch_encoding_size=args.batch_encoding_size
    )