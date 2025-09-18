#!/usr/bin/env python

"""teleop_with_recording_lerobot_videos.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording
in the LeRobot dataset format, saving camera data as MP4 videos. Gripper functionality has been removed.
"""

import json
import time
import argparse
import os
import numpy as np
import spdlog
from datetime import datetime
from pathlib import Path
import torch
import PIL.Image
from typing import Dict, List
import shutil
import tempfile

# Import utility methods
from utility import quat2eulerZYX, list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion
from quest_receive import QuestTeleop

# Import Realsense python libraries
from realsense_record import RealSenseModule, get_rgbd, CameraConfig

# Import LeRobot utilities
from lerobot.datasets.utils import (
    create_empty_dataset_info,
    write_json,
    validate_frame,
    get_hf_features_from_features,
    DEFAULT_IMAGE_PATH,
)
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.datasets.video_utils import encode_video_frames, get_video_info, get_video_duration_in_s

class LeRobotTrajectoryRecorder:
    def __init__(self, repo_id: str, output_dir: str, camera_config: CameraConfig, fps: int = 30):
        """Initialize the LeRobot dataset recorder with video support."""
        # Initialize cameras
        self.cameras = RealSenseModule(camera_config)
        self.is_recording = True
        self.episode_index = 0

        self.logger = spdlog.ConsoleLogger("Recorder")
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.num_cameras = len(self.cameras.serial_numbers)
        self.logger.info(f"Initialized {self.num_cameras} cameras")

        # Define features for the LeRobot dataset
        self.features = {
            "index": {"dtype": "int32", "shape": [], "_index": True},
            "episode_index": {"dtype": "int32", "shape": [], "_index": True},
            "timestamp": {"dtype": "float32", "shape": []},
            "task_index": {"dtype": "int32", "shape": []},
            "q": {"dtype": "float32", "shape": [7], "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]},
            "theta": {"dtype": "float32", "shape": [7], "names": ["theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "theta7"]},
            "dq": {"dtype": "float32", "shape": [7], "names": ["dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7"]},
            "dtheta": {"dtype": "float32", "shape": [7], "names": ["dtheta1", "dtheta2", "dtheta3", "dtheta4", "dtheta5", "dtheta6", "dtheta7"]},
            "tau": {"dtype": "float32", "shape": [7], "names": ["tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]},
            "tau_des": {"dtype": "float32", "shape": [7], "names": ["tau_des1", "tau_des2", "tau_des3", "tau_des4", "tau_des5", "tau_des6", "tau_des7"]},
            "tau_dot": {"dtype": "float32", "shape": [7], "names": ["tau_dot1", "tau_dot2", "tau_dot3", "tau_dot4", "tau_dot5", "tau_dot6", "tau_dot7"]},
            "tau_ext": {"dtype": "float32", "shape": [7], "names": ["tau_ext1", "tau_ext2", "tau_ext3", "tau_ext4", "tau_ext5", "tau_ext6", "tau_ext7"]},
            "tcp_pose": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
            "tcp_velocity": {"dtype": "float32", "shape": [6], "names": ["vx", "vy", "vz", "wx", "wy", "wz"]},
            "flange_pose": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
            "ft_sensor_raw": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
            "f_ext_tcp_frame": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
            "f_ext_base_frame": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
            "action": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        }
        for i in range(self.num_cameras):
            self.features[f"observation.image.cam{i+1}"] = {
                "dtype": "video",
                "shape": [camera_config.rgb_size[1], camera_config.rgb_size[0], 3]
            }

        # Initialize LeRobot dataset with video support
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=self.features,
            root=output_dir,
            robot_type="Flexiv Rizon 4s",
            use_videos=True,  # Save as videos
            image_writer_processes=4,
            image_writer_threads=4,
            video_backend="pyav",
            batch_encoding_size=1
        )

    def add_state(self, robot_states, quest_input=None):
        """Add state data for one timestep to the episode buffer."""
        if not self.is_recording:
            return

        try:
            frame = {
                "timestamp": time.time(),
                "q": np.array(robot_states.q, dtype=np.float32),
                "theta": np.array(robot_states.theta, dtype=np.float32),
                "dq": np.array(robot_states.dq, dtype=np.float32),
                "dtheta": np.array(robot_states.dtheta, dtype=np.float32),
                "tau": np.array(robot_states.tau, dtype=np.float32),
                "tau_des": np.array(robot_states.tau_des, dtype=np.float32),
                "tau_dot": np.array(robot_states.tau_dot, dtype=np.float32),
                "tau_ext": np.array(robot_states.tau_ext, dtype=np.float32),
                "tcp_pose": np.array(robot_states.tcp_pose, dtype=np.float32),
                "tcp_velocity": np.array(robot_states.tcp_vel, dtype=np.float32),
                "flange_pose": np.array(robot_states.flange_pose, dtype=np.float32),
                "ft_sensor_raw": np.array(robot_states.ft_sensor_raw, dtype=np.float32),
                "f_ext_tcp_frame": np.array(robot_states.ext_wrench_in_tcp, dtype=np.float32),
                "f_ext_base_frame": np.array(robot_states.ext_wrench_in_world, dtype=np.float32),
            }

            # Add camera images (stored temporarily as PNGs)
            camera_data = get_rgbd(self.cameras)
            if len(camera_data) != self.num_cameras:
                self.logger.error(f"Expected {self.num_cameras} camera feeds, but got {len(camera_data)}")
                return
            for i, (image, _, _) in enumerate(camera_data):
                frame[f"observation.image.cam{i+1}"] = np.array(image, dtype=np.uint8)

            # Add action (will be updated by add_action)
            frame["action"] = np.zeros(7, dtype=np.float32)  # Placeholder
            frame["task"] = "teleoperation"  # Default task name, updated later

            # Validate and add frame to episode buffer
            validate_frame(frame, self.features)
            self.dataset.add_frame(frame)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")

    def add_action(self, offset_pos, offset_quat):
        """Update the action in the last frame of the episode buffer."""
        if not self.is_recording or self.dataset.episode_buffer is None:
            return

        try:
            action = np.array([
                offset_pos[0], offset_pos[1], offset_pos[2],
                offset_quat.w, offset_quat.x, offset_quat.y, offset_quat.z
            ], dtype=np.float32)
            # Update the action in the last frame
            if self.dataset.episode_buffer["size"] > 0:
                self.dataset.episode_buffer["action"][-1] = action
        except Exception as e:
            self.logger.error(f"Error adding action data: {str(e)}")

    def save_trajectory(self, task: str):
        """Save the current episode to the LeRobot dataset, encoding images as videos."""
        if self.dataset.episode_buffer is None or self.dataset.episode_buffer["size"] == 0:
            self.logger.warn("No data to save in episode buffer")
            return

        self.logger.info(f"Saving episode for task: {task}")
        self.dataset.episode_buffer["task"] = [task] * self.dataset.episode_buffer["size"]
        self.dataset.save_episode()
        self.episode_index += 1
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(episode_index=self.episode_index)
        self.logger.info(f"Episode saved, new episode index: {self.episode_index}")

    def stop_recording(self):
        """Stop recording and clean up resources."""
        self.is_recording = False
        self.dataset.stop_image_writer()
        self.cameras.cleanup()
        self.logger.info("Recording stopped and resources cleaned up")

class LeRobotDataset:
    @classmethod
    def create(cls, repo_id: str, fps: int, features: Dict, root: str, robot_type: str, use_videos: bool,
               image_writer_processes: int = 0, image_writer_threads: int = 0, video_backend: str = "pyav",
               batch_encoding_size: int = 1):
        """Create a LeRobot Dataset from scratch for recording data."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) / repo_id
        obj.fps = fps
        obj.features = features
        obj.robot_type = robot_type
        obj.use_videos = use_videos
        obj.video_backend = video_backend
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        # Initialize metadata
        obj.root.mkdir(parents=True, exist_ok=True)
        obj.info = create_empty_dataset_info(
            codebase_version="v3.0",
            fps=fps,
            features=features,
            use_videos=use_videos,
            robot_type=robot_type
        )
        obj.info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        write_json(obj.info, obj.root / "meta/info.json")

        # Initialize dataset
        obj.hf_dataset = cls.create_hf_dataset(features)
        obj.episode_buffer = cls.create_episode_buffer(0)
        obj.image_writer = None
        if image_writer_processes or image_writer_threads:
            obj.image_writer = AsyncImageWriter(
                num_processes=image_writer_processes,
                num_threads=image_writer_threads
            )
        obj.total_episodes = 0
        obj.total_frames = 0
        obj.stats = None
        return obj

    @staticmethod
    def create_hf_dataset(features):
        """Create an empty Hugging Face dataset."""
        from datasets import Dataset, Features
        hf_features = get_hf_features_from_features(features)
        ft_dict = {col: [] for col in hf_features}
        dataset = Dataset.from_dict(ft_dict, features=hf_features, split="train")
        dataset.set_transform(lambda x: {k: torch.tensor(v) if isinstance(v, (list, np.ndarray)) else v for k, v in x.items()})
        return dataset

    @staticmethod
    def create_episode_buffer(episode_index: int):
        """Create an episode buffer for storing frames."""
        ep_buffer = {
            "size": 0,
            "task": [],
            "episode_index": episode_index,
            "index": [],
            "timestamp": [],
            "task_index": [],
        }
        for key in ep_buffer:
            if key not in ["size", "task", "episode_index"]:
                ep_buffer[key] = []
        return ep_buffer

    def add_frame(self, frame: Dict):
        """Add a frame to the episode buffer, storing images temporarily."""
        frame["episode_index"] = self.episode_buffer["episode_index"]
        frame["index"] = self.total_frames + self.episode_buffer["size"]
        validate_frame(frame, self.features)
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp")
        task = frame.pop("task")

        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)
        self.episode_buffer["frame_index"].append(frame_index)
        for key in frame:
            if self.features[key]["dtype"] == "video":
                # Store images temporarily for video encoding
                img_path = self.root / f"temp_images/{key}/episode_{self.episode_buffer['episode_index']:03d}/frame_{frame_index:06d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(frame[key], (torch.Tensor, np.ndarray)):
                    image = np.array(frame[key], dtype=np.uint8)
                elif isinstance(frame[key], PIL.Image.Image):
                    image = frame[key]
                else:
                    raise ValueError(f"Unsupported image type for {key}: {type(frame[key])}")
                if self.image_writer:
                    self.image_writer.save_image(image, img_path)
                else:
                    write_image(image, img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(np.array(frame[key], dtype=np.float32))
        self.episode_buffer["size"] += 1

    def save_episode(self):
        """Save the current episode to disk, encoding images as videos."""

        from datasets import Dataset
        import pandas as pd
        episode_buffer = self.episode_buffer
        episode_length = episode_buffer["size"]
        tasks = list(set(episode_buffer.pop("task")))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.total_frames, self.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full(episode_length, episode_index)
        episode_buffer["task_index"] = np.array([0] * episode_length)  # Single task for simplicity

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] == "video":
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to finish
        if self.image_writer:
            self.image_writer.wait_until_done()

        # Encode videos for each camera
        video_metadata = {}
        for key in [k for k, v in self.features.items() if v["dtype"] == "video"]:
            video_metadata.update(self._save_episode_video(key, episode_index))

        # Compute episode statistics
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        # Save episode data
        ep_dict = {key: episode_buffer[key] for key in self.features if key in episode_buffer and self.features[key]["dtype"] != "video"}
        ep_dataset = Dataset.from_dict(ep_dict, features=get_hf_features_from_features(self.features))
        ep_size_in_mb = len(ep_dataset) * sum(np.prod(v["shape"]) * 4 for k, v in self.features.items() if v["dtype"] != "video") / 1e6

        chunk_idx, file_idx = 0, 0
        latest_num_frames = 0
        if self.total_episodes > 0:
            chunk_idx = self.total_episodes // self.info.get("chunks_size", 1000)
            file_idx = self.total_episodes % self.info.get("chunks_size", 1000)
            latest_path = self.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
            if latest_path.exists():
                latest_df = pd.read_parquet(latest_path)
                latest_num_frames = len(latest_df)
                df = pd.concat([latest_df, pd.DataFrame(ep_dict)], ignore_index=True)
                del latest_df
            else:
                df = pd.DataFrame(ep_dict)
        else:
            df = pd.DataFrame(ep_dict)

        path = self.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

        # Update episode metadata
        ep_metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": latest_num_frames,
            "dataset_to_index": latest_num_frames + episode_length,
            **video_metadata
        }

        # Save episode metadata
        ep_metadata_dict = {
            "episode_index": episode_index,
            "tasks": tasks,
            "length": episode_length,
            **ep_metadata
        }
        ep_metadata_df = pd.DataFrame([ep_metadata_dict])
        ep_metadata_path = self.root / f"meta/episodes/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        ep_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if ep_metadata_path.exists():
            existing_df = pd.read_parquet(ep_metadata_path)
            ep_metadata_df = pd.concat([existing_df, ep_metadata_df], ignore_index=True)
        ep_metadata_df.to_parquet(ep_metadata_path)

        # Update dataset metadata
        self.total_episodes += 1
        self.total_frames += episode_length
        self.info["total_episodes"] = self.total_episodes
        self.info["total_frames"] = self.total_frames
        self.info["total_tasks"] = 1
        self.info["splits"] = {"train": f"0:{self.total_episodes}"}
        if episode_index == 0:
            for key in [k for k, v in self.features.items() if v["dtype"] == "video"]:
                video_path = self.root / self.info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
                self.info["features"][key]["info"] = get_video_info(video_path)
        write_json(self.info, self.root / "meta/info.json")

        # Save tasks
        tasks_df = pd.DataFrame({"task_index": [0]}, index=tasks)
        tasks_df.to_parquet(self.root / "meta/tasks.parquet")

        # Save stats
        from lerobot.datasets.utils import write_stats, aggregate_stats
        self.stats = aggregate_stats([self.stats, ep_stats]) if hasattr(self, "stats") else ep_stats
        write_stats(self.stats, self.root)

        # Update Hugging Face dataset
        self.hf_dataset = self.create_hf_dataset(self.features)

        # Clear temporary images
        for key in [k for k, v in self.features.items() if v["dtype"] == "video"]:
            temp_img_dir = self.root / f"temp_images/{key}/episode_{episode_index:03d}"
            if temp_img_dir.exists():
                shutil.rmtree(temp_img_dir)

    def _save_episode_video(self, video_key: str, episode_index: int):
        """Encode temporary images into an MP4 video."""
        temp_img_dir = self.root / f"temp_images/{video_key}/episode_{episode_index:03d}"
        if not temp_img_dir.exists():
            raise ValueError(f"Temporary image directory {temp_img_dir} does not exist")

        # Encode video
        temp_video_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        encode_video_frames(temp_img_dir, temp_video_path, self.fps, overwrite=True)
        ep_duration_in_s = get_video_duration_in_s(temp_video_path)

        # Determine chunk and file indices
        chunk_idx, file_idx = 0, 0
        latest_duration_in_s = 0.0
        if self.total_episodes > 0:
            chunk_idx = self.total_episodes // self.info.get("chunks_size", 1000)
            file_idx = self.total_episodes % self.info.get("chunks_size", 1000)
            latest_path = self.root / self.info["video_path"].format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
            if latest_path.exists():
                latest_duration_in_s = get_video_duration_in_s(latest_path)

        # Move video to final location
        final_path = self.root / self.info["video_path"].format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_video_path), str(final_path))

        # Clean up temporary video directory
        shutil.rmtree(temp_video_path.parent)

        return {
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }

    def stop_image_writer(self):
        """Stop the asynchronous image writer."""
        if self.image_writer:
            self.image_writer.stop()
            self.image_writer = None

def get_cur_pose(robot):
    """Get current robot pose."""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    return robot_states, current_tcp_pos, current_tcp_quat

def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, save_path="./data"):
    """Main function for teleoperation with recording in LeRobot format with videos."""
    logger = spdlog.ConsoleLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording in LeRobot format with videos.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = f"teleop_trajectory_{timestamp}"

    # Initialize CameraConfig
    camera_config = CameraConfig(
        real_time_view=True,
        rgb_size=(rgb_width, rgb_height),
        depth_size=(rgb_width, rgb_height),
        fps=fps,
        save_path=save_path,
        save_freq=frequency
    )

    recorder = LeRobotTrajectoryRecorder(repo_id, path, camera_config, fps)
    quest_controller = QuestTeleop()

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

        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(1)

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warn("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        while not robot.primitive_states()["terminated"]:
            time.sleep(1)
        logger.info("Sensor zeroing complete")

        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info(f"Starting teleoperation, recording to: {path}/{repo_id}")

        last_input = None
        frame_cnt = 0
        last_robot_states, last_tcp_pos, last_tcp_quat = get_cur_pose(robot)

        while True:
            robot_states = robot.states()
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

            recorder.add_state(robot_states)
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

            # Initialize a flag to track if this is the first iteration
            is_first_iteration = True

            if current_input.get('rightHand', 0) > 0.5:
                if is_first_iteration or last_input.get('rightHand', 0) <= 0.5:
                    if not is_first_iteration:
                        start_tcp_pos = current_tcp_pos
                        start_tcp_quat = current_tcp_quat
                        start_input_pos = current_input_pos
                        start_input_quat = current_input_quat
                is_first_iteration = False

                offset_pos = current_input_pos - start_input_pos
                offset_quat = quaternion.quaternion.inverse(start_input_quat) * current_input_quat
                pos = start_tcp_pos + offset_pos
                quat = start_tcp_quat * offset_quat
                robot.SendCartesianMotionForce([*pos, quat.w, quat.x, quat.y, quat.z], [0.0] * 6)
                recorder.add_action(pos, quat)
            else:
                recorder.add_action(current_tcp_pos, current_tcp_quat)

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
        recorder.save_trajectory(task)
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        while robot.busy():
            time.sleep(0.02)
        recorder.stop_recording()
        logger.info(f"Trajectory saved to {path}/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording in LeRobot format with videos")
    parser.add_argument("--path", type=str, default="./teleop_recordings/", help="Path to save recordings")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="teleoperation", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=480, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--save_path", type=str, default="./data", help="Path to save RGB-D data")
    args = parser.parse_args()
    main(
        task=args.task,
        path=args.path,
        frequency=args.frequency,
        rgb_width=args.rgb_width,
        rgb_height=args.rgb_height,
        fps=args.fps,
        save_path=args.save_path
    )