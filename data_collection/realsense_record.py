import math
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class RealSenseModule:
    def __init__(self, real_time_view=False, rgb_size=[640, 480]):
        self.state = AppState()

        # Create a pipeline instance
        self.pipeline = rs.pipeline()

        # Configure camera
        self.config = rs.config()

        # Get connected devices
        self.ctx = rs.context()
        self.devices = list(self.ctx.query_devices())

        if len(self.devices) < 1:
            raise RuntimeError("No RealSense camera detected")

        # Get camera serial number
        self.serial = self.devices[0].get_info(rs.camera_info.serial_number)

        # Configure camera
        self.config.enable_device(self.serial)

        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, rgb_size[0], rgb_size[1], rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        # Get camera profile
        self.profile = self.pipeline.get_active_profile()

        # Set up aligner
        self.align = rs.align(rs.stream.color)

        # Get depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        if real_time_view:
            cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)

    def get_camera_intrinsics(self, profile):
        color_stream = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_stream.get_intrinsics()

        mtx = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy]
        camIntrinsics = np.array([[mtx[4], 0, mtx[2]],
                                 [0, mtx[5], mtx[3]],
                                 [0, 0, 1.]])

        return camIntrinsics, intrinsics.coeffs

    def get_data(self):
        """Get data from single camera"""
        step = 0
        while True:
            # Wait for camera frames
            frames = self.pipeline.wait_for_frames()

            # Align depth and color frames
            aligned_frames = self.align.process(frames)

            # Get depth and color frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not all([depth_frame, color_frame]):
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Get camera parameters
            camIntrinsics, distCoeffs = self.get_camera_intrinsics(self.profile)

            break

        return color_image, depth_image, camIntrinsics, distCoeffs

def _init_rs_camera(real_time_view=False):
    rs_module = RealSenseModule(real_time_view=real_time_view)
    return rs_module

def save_rgbd_seqs(rs_module, save_path='./force_feedback/replay_data/1/rgbd', saving_freq=10):
    view_step = 0

    os.makedirs(save_path, exist_ok=True)

    timesleep = 1. / saving_freq

    while True:
        try:
            # Get data from camera
            color_image, depth_image, camIntrinsics, distCoeffs = rs_module.get_data()

            # Save data
            np.save(os.path.join(save_path, f'color_image_{view_step}.npy'), color_image)
            np.save(os.path.join(save_path, f'depth_image_{view_step}.npy'), depth_image)
            np.save(os.path.join(save_path, f'camIntrinsics.npy'), camIntrinsics)
            cv2.imwrite(os.path.join(save_path, f'color_image_{view_step}.jpg'), color_image)

            view_step += 1
            print('view_step: ', view_step)

            time.sleep(timesleep)

        except KeyboardInterrupt:
            print("Exiting...")
            break

def get_rgbd(rs_module):
    # Get data from camera
    color_image, depth_image, camIntrinsics, distCoeffs = rs_module.get_data()
    return color_image, depth_image, camIntrinsics

if __name__ == '__main__':
    camera = _init_rs_camera()

    image, depth, intrinsics = get_rgbd(camera)

    cv2.imshow('Camera Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save_rgbd_seqs(camera, './test_camera', 30)