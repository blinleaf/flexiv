#!/usr/bin/env python

"""flexiv_replay_from_lerobot.py

This script replays a recorded episode from a LeRobot dataset to a Flexiv robot.
It loads the dataset, selects an episode, and sends the actions (TCP pose and gripper) to the robot
in a synchronized manner based on the dataset FPS.
"""

import argparse
import time
import os
import numpy as np
import spdlog
from datetime import datetime
from pathlib import Path
import torch
from typing import Dict, List
import quaternion

# Import Flexiv RDK python libraries
import flexivrdk

# Import LeRobot utilities
from lerobot import LeRobotDataset
from lerobot.datasets.utils import load_info

def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose."""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    gripper_states = gripper.states()
    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states

def main(repo_id: str, root: str, episode_index: int, speed_scale: float = 1.0, fps: int = 30, path: str = "./"):
    """Main function for replaying LeRobot dataset to Flexiv robot."""
    logger = spdlog.ConsoleLogger("Replay")
    logger.info(f"Replaying episode {episode_index} from LeRobot dataset '{repo_id}' to Flexiv robot.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)

    # Initialize LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode_index],  # Load only the specified episode
        download_videos=False,  # No need for videos during replay
    )
    logger.info(f"Loaded dataset with {len(dataset)} frames at {dataset.fps} FPS.")

    # Load info to get FPS if not specified
    info = load_info(Path(root) / repo_id)
    actual_fps = info["fps"]
    logger.info(f"Dataset FPS: {actual_fps}, using {fps} for replay timing.")

    try:
        # RDK Initialization
        robot = flexivrdk.Robot("Rizon 4s-063034")
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
        gripper.Enable("Flexiv-GN01")
        logger.info("Opening gripper")
        gripper.Move(0.1, 0.1, 20)
        time.sleep(1)

        # Move to home position
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(1)
        logger.info("Moved to home position")

        # Zero FT sensor
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warn("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        while not robot.primitive_states()["terminated"]:
            time.sleep(1)
        logger.info("Sensor zeroing complete")

        # Switch to Cartesian Motion Force mode for replay
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info("Starting replay in Cartesian Motion Force mode")

        # Get the episode data
        episode_length = dataset.num_frames
        logger.info(f"Replaying {episode_length} frames...")

        # Optional: Move to starting pose of the episode (first action pose)
        first_frame = dataset[0]
        start_pos = first_frame["action"][:3].numpy()
        start_quat = quaternion.quaternion(*first_frame["action"][3:7].numpy())
        logger.info(f"Moving to starting pose: pos={start_pos}, quat={start_quat}")
        robot.SendCartesianMotionForce([*start_pos, start_quat.w, start_quat.x, start_quat.y, start_quat.z], [0.0] * 6)
        time.sleep(2)  # Wait for robot to reach starting pose

        # Replay loop
        start_time = time.time()
        for frame_idx in range(episode_length):
            # Get the frame
            frame = dataset[frame_idx]
            action = frame["action"].numpy()  # [x, y, z, qw, qx, qy, qz, gripper_close]
            pos = action[:3]
            quat = quaternion.quaternion(action[3], action[4], action[5], action[6])
            gripper_close = action[7]

            # Scale speed if needed (but since it's position-based, speed_scale might affect timing)
            # For position control, timing controls speed
            target_time = frame_idx / fps
            current_time = time.time() - start_time
            if current_time < target_time:
                time.sleep(target_time - current_time)

            # Send pose (with zero force for position control)
            robot.SendCartesianMotionForce([*pos, quat.w, quat.x, quat.y, quat.z], [0.0] * 6)

            # Move gripper
            gripper.Move(gripper_close, 0.1, 20)

            if frame_idx % 30 == 0:  # Log every 30 frames
                logger.info(f"Replayed frame {frame_idx}/{episode_length}: pos={pos}, gripper={gripper_close:.3f}")

        logger.info("Replay completed")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during replay: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Return to home
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(0.02)
        logger.info("Returned to home position")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay LeRobot dataset to Flexiv robot")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id")
    parser.add_argument("--root", type=str, default=None, help="Local root path for dataset (default: ~/.cache/huggingface/lerobot)")
    parser.add_argument("--episode_index", type=int, required=True, help="Episode index to replay")
    parser.add_argument("--speed_scale", type=float, default=1.0, help="Speed scale for replay (affects timing)")
    parser.add_argument("--fps", type=int, default=30, help="Replay FPS (default: 30)")
    parser.add_argument("--path", type=str, default="./", help="Working path")
    args = parser.parse_args()
    main(
        repo_id=args.repo_id,
        root=args.root,
        episode_index=args.episode_index,
        speed_scale=args.speed_scale,
        fps=args.fps,
        path=args.path
    )