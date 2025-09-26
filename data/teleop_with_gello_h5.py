#!/usr/bin/env python
import sys
import h5py
import time
import threading
import spdlog
import flexivrdk
import numpy as np
import quaternion
from Piper.piper_arm import PiperArm
from scipy.spatial.transform import Rotation as R
import math
from transforms3d.euler import euler2quat
from d435_camera import Camera
from realsense_record import RealSenseModule, get_rgbd, CameraConfig
from typing import Tuple, List, Optional
from remote_communication import RemoteCommunication


def wrap_angle_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    if a <= -180.0:
        a += 360.0
    return a

def get_rgbd(rs_module: RealSenseModule) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get RGB-D data from all cameras"""
    data = rs_module.get_data()
    return [(color_img, depth_img, cam_intrinsics) for color_img, depth_img, cam_intrinsics, _ in data]


class NeuralNetworkController:
    """神经网络控制器（占位符）"""
    
    def __init__(self):
        self.logger = spdlog.ConsoleLogger("NeuralNetwork")
        self.logger.info("神经网络控制器初始化")
        
    def get_action(self, current_tcp_pos_xyz, current_tcp_quat, camera_images):
        """
        获取神经网络预测的动作
        
        Args:
            current_tcp_pos_xyz: 当前TCP位置
            current_tcp_quat: 当前TCP姿态
            camera_images: 相机图像数据
            
        Returns:
            tuple: (delta_pose_xyz, delta_pose_quat, gripper_value)
        """
        # TODO: 实现神经网络推理
        # 这里是占位符，返回零动作
        delta_pose_xyz = np.array([0.0, 0.0, 0.0])
        delta_pose_quat = np.array([0.0, 0.0, 0.0])
        gripper_value = 0.05
        
        self.logger.info("神经网络控制器提供动作")
        return delta_pose_xyz, delta_pose_quat, gripper_value


class Mode3Controller:
    """模式3控制器（占位符）"""
    
    def __init__(self):
        self.logger = spdlog.ConsoleLogger("Mode3")
        self.logger.info("模式3控制器初始化")
        
    def get_action(self, current_tcp_pos_xyz, current_tcp_quat, camera_images):
        """
        获取模式3的动作
        
        Args:
            current_tcp_pos_xyz: 当前TCP位置
            current_tcp_quat: 当前TCP姿态
            camera_images: 相机图像数据
            
        Returns:
            tuple: (delta_pose_xyz, delta_pose_quat, gripper_value)
        """
        # TODO: 实现模式3逻辑
        # 这里是占位符，返回零动作
        delta_pose_xyz = np.array([0.0, 0.0, 0.0])
        delta_pose_quat = np.array([0.0, 0.0, 0.0])
        gripper_value = 0.05
        
        self.logger.info("模式3控制器提供动作")
        return delta_pose_xyz, delta_pose_quat, gripper_value


class ClientTimeoutMonitor:
    """客户端超时监控器"""
    
    def __init__(self, timeout_seconds=3.0):
        self.timeout_seconds = timeout_seconds
        self.last_client_data_time = None
        self.client_connected = False
        self.logger = spdlog.ConsoleLogger("ClientTimeout")
        
    def update_client_data(self, pose_data):
        """更新客户端数据时间戳"""
        if pose_data is not None:
            self.last_client_data_time = time.time()
            if not self.client_connected:
                self.client_connected = True
                self.logger.info("客户端连接已恢复")
        
    def is_client_timeout(self):
        """检查客户端是否超时"""
        if self.last_client_data_time is None:
            return True  # 从未收到过数据，认为超时
            
        time_since_last_data = time.time() - self.last_client_data_time
        is_timeout = time_since_last_data > self.timeout_seconds
        
        if is_timeout and self.client_connected:
            self.client_connected = False
            self.logger.warn(f"客户端超时 ({time_since_last_data:.1f}s > {self.timeout_seconds}s)，切换到mode3")
            
        return is_timeout
    
    def get_status_info(self):
        """获取状态信息"""
        if self.last_client_data_time is None:
            return "客户端: 未连接"
        
        time_since_last = time.time() - self.last_client_data_time
        if self.is_client_timeout():
            return f"客户端: 超时 ({time_since_last:.1f}s)"
        else:
            return f"客户端: 正常 ({time_since_last:.1f}s)"


class PiperSubArmThread(threading.Thread):
    """子机械臂线程基类"""
    def __init__(self, arm_id, can_interface, master_state_shared):
        super().__init__()
        self.arm_id = arm_id
        self.can_interface = can_interface
        self.master_state_shared = master_state_shared  # 共享的主机械臂状态
        self.running = False
        self.piper_arm = None
        self.logger = spdlog.ConsoleLogger(f"PiperSubArm-{arm_id}")
        
    def run(self):
        return
        """线程主循环"""
        try:
            self.logger.info(f"子机械臂 {self.arm_id} 线程启动")
            self.init_piper_arm()
            self.running = True
            
            while self.running:
                # 获取主机械臂当前状态
                with self.master_state_shared['lock']:
                    master_pos = self.master_state_shared.get('current_tcp_pos_xyz', None)
                    master_quat = self.master_state_shared.get('current_tcp_quat', None)
                    control_mode = self.master_state_shared.get('control_mode', 'mode3')
                
                if master_pos is not None and master_quat is not None:
                    # 执行具体的动作逻辑（子类实现）
                    self.execute_action_logic(master_pos, master_quat, control_mode)
                
                time.sleep(0.1)  # 100ms控制周期
                
        except Exception as e:
            self.logger.error(f"子机械臂 {self.arm_id} 线程错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def init_piper_arm(self):
        """初始化Piper机械臂 - 通用启动代码"""
        try:
            self.logger.info(f"初始化子机械臂 {self.arm_id} (CAN: {self.can_interface})")
            self.piper_arm = PiperArm(can=self.can_interface)
            # TODO: 添加具体的Piper机械臂初始化代码
            self.logger.info(f"子机械臂 {self.arm_id} 初始化完成")
        except Exception as e:
            self.logger.error(f"子机械臂 {self.arm_id} 初始化失败: {str(e)}")
            raise
    
    def execute_action_logic(self, master_pos, master_quat, control_mode):
        """执行具体的动作逻辑 - 子类需要重写此方法"""
        # TODO: 子类实现具体的动作逻辑，可以根据control_mode调整行为
        pass
    
    def stop(self):
        """停止线程"""
        self.running = False
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.piper_arm:
                # TODO: 添加Piper机械臂清理代码
                pass
            self.logger.info(f"子机械臂 {self.arm_id} 清理完成")
        except Exception as e:
            self.logger.error(f"子机械臂 {self.arm_id} 清理错误: {str(e)}")


class PiperSubArm1(PiperSubArmThread):
    """第一个子机械臂 - 可以有自己的特殊逻辑"""
    def __init__(self, master_state_shared):
        super().__init__("Arm1", "can1", master_state_shared)
    
    def execute_action_logic(self, master_pos, master_quat, control_mode):
        """第一个子机械臂的动作逻辑"""
        # TODO: 实现第一个子机械臂的具体动作逻辑
        # 可以根据control_mode调整行为
        if control_mode == "remote":
            # 远程控制模式下的行为
            pass
        elif control_mode == "neural":
            # 神经网络模式下的行为
            pass
        elif control_mode == "mode3":
            # 模式3下的行为
            pass


class PiperSubArm2(PiperSubArmThread):
    """第二个子机械臂 - 可以有自己的特殊逻辑"""
    def __init__(self, master_state_shared):
        super().__init__("Arm2", "can2", master_state_shared)
    
    def execute_action_logic(self, master_pos, master_quat, control_mode):
        """第二个子机械臂的动作逻辑"""
        # TODO: 实现第二个子机械臂的具体动作逻辑
        # 可以根据control_mode调整行为
        if control_mode == "remote":
            # 远程控制模式下的行为
            pass
        elif control_mode == "neural":
            # 神经网络模式下的行为
            pass
        elif control_mode == "mode3":
            # 模式3下的行为
            pass


class Zty_TrajectoryRecorder:
    def __init__(self, task_config, use_remote=True, remote_ip="192.168.1.100", image_quality=50, client_timeout=3.0):
        self.task_config = task_config
        self.task = task_config['task_name']
        self.camera_config = task_config['camera_config']
        self.camera_num = self.camera_config['camera_num']
        self.camera_info = self.camera_config['camera_info']
        self.output_file = task_config['output_file']
        
        # 远程通信设置
        self.use_remote = use_remote
        self.remote_ip = remote_ip
        self.remote_comm = None
        
        # 客户端超时监控
        self.timeout_monitor = ClientTimeoutMonitor(timeout_seconds=client_timeout)
        
        # 共享的主机械臂状态变量
        self.master_state_shared = {
            'current_tcp_pos_xyz': None,
            'current_tcp_quat': None,
            'control_mode': 'mode3',  # 默认模式改为mode3
            'lock': threading.Lock()  # 用于线程安全
        }
        
        # 子机械臂线程
        self.sub_arm1 = None
        self.sub_arm2 = None
        
        # 多模式控制器
        self.neural_controller = NeuralNetworkController()
        self.mode3_controller = Mode3Controller()
        
        # self.camera_moudle = Camera(camera_device_mapping = self.camera_info)
        camera_config = CameraConfig(
            real_time_view=False,
            rgb_size=(640, 480),
            depth_size=(640, 480),
            fps=30,
            save_freq=30
        )
        self.cameras = RealSenseModule(camera_config)
        self.num_cameras = len(self.cameras.serial_numbers)
        
        # 根据是否使用远程通信决定是否初始化从臂
        if not self.use_remote:
            self.master_piper = PiperArm(can='can0')
        else:
            self.master_piper = None
            # 初始化远程通信，设置图像压缩质量
            self.remote_comm = RemoteCommunication(image_quality=image_quality)
            
        self.logger = spdlog.ConsoleLogger("Program Starting...")
        self.mode = flexivrdk.Mode
        self.flexiv_init_pose = [
            0.5953401923179626, 
            -0.12538231909275055, 
            0.33247876167297363, 
            0.04877982661128044, 
            0.08595447242259979, 
            0.9943479299545288, 
            0.03878817334771156
        ]
        self.flexiv_init()
        self.trajRecorder_init()
        
        # 启动子机械臂线程
        self.start_sub_arms()

    def start_sub_arms(self):
        """启动两个子机械臂线程"""
        try:
            self.logger.info("启动子机械臂线程...")
            
            # 创建并启动第一个子机械臂线程
            self.sub_arm1 = PiperSubArm1(self.master_state_shared)
            self.sub_arm1.start()
            
            # 创建并启动第二个子机械臂线程
            self.sub_arm2 = PiperSubArm2(self.master_state_shared)
            self.sub_arm2.start()
            
            self.logger.info("子机械臂线程启动完成")
            
        except Exception as e:
            self.logger.error(f"启动子机械臂线程失败: {str(e)}")
            raise

    def stop_sub_arms(self):
        """停止子机械臂线程"""
        try:
            self.logger.info("停止子机械臂线程...")
            
            if self.sub_arm1:
                self.sub_arm1.stop()
                self.sub_arm1.join(timeout=2.0)
                
            if self.sub_arm2:
                self.sub_arm2.stop()
                self.sub_arm2.join(timeout=2.0)
                
            self.logger.info("子机械臂线程停止完成")
            
        except Exception as e:
            self.logger.error(f"停止子机械臂线程失败: {str(e)}")

    def trajRecorder_init(self,):
        self.timestamps = []
        self.tcp_pose_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        self.camera_images_list = {
            v: [] for v in self.camera_info.values()
        }

    def add_state(self, robot_states, gripper_states):
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

        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")

    def flexiv_init(self,):
        self.robot = flexivrdk.Robot('Rizon 4s-063034')

        if self.robot.fault():
            self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
            self.robot.ClearFault()
            time.sleep(2)
            if not self.robot.ClearFault():
                self.logger.error("Fault cannot be cleared, exiting ...")
                return
            self.logger.info("Fault on the connected robot is cleared")

        self.logger.info("Enabling robot ...")
        self.robot.Enable()
        seconds_waited = 0
        while not self.robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                self.logger.warn("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                return
        self.logger.info("Robot operational")

        self.logger.info(f"Enabling gripper 'Flexiv-GN01'")
        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper.Enable("Flexiv-GN01")

        # Open Gripper First
        self.logger.info("Opening gripper...")
        self.gripper.Move(0.1, 0.2, 20)
        while self.robot.busy():
            time.sleep(1)

        self.robot.SwitchMode(self.mode.NRT_PLAN_EXECUTION)
        self.robot.ExecutePlan("PLAN-Home")
        while self.robot.busy():
            time.sleep(1)

        # 
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        self.logger.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(1)
        self.logger.info("Sensor zeroing complete")

        # 
        # robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        # self.robot.SendCartesianMotionForce(self.init_pose, [0.0] * 6, max_linear_vel = 0.09, max_angular_vel = 0.3,)
    
    def get_action_from_mode(self, control_mode, current_tcp_pos_xyz, current_tcp_quat, pose_data):
        """
        根据控制模式获取动作指令
        
        Args:
            control_mode: 控制模式 ('remote', 'neural', 'mode3')
            current_tcp_pos_xyz: 当前TCP位置
            current_tcp_quat: 当前TCP姿态
            pose_data: 远程通信数据（可能为None）
            
        Returns:
            tuple: (delta_pose_xyz, delta_pose_quat, gripper_value)
        """
        if control_mode == "remote":
            # 远程控制模式
            if pose_data is not None:
                end_effector_pos = np.array(pose_data['end_effector_pos'])
                end_effector_ori = np.array(pose_data['end_effector_ori'])
                leader_gripper_pos = pose_data['leader_gripper_pos']

                return end_effector_pos,  end_effector_ori, leader_gripper_pos

                # return master_delta_pose_xyz, master_delta_pose_quat, master_gripper
                
            # else:
            #     # 使用本地从臂
            #     if self.master_piper:
            #         master_delta = self.master_piper.get_delta_pose()
            #         master_gripper = (self.master_piper.get_arm_eepose()[-1] / 80000) / 10
            #         master_gripper = max(0, min(0.09, master_gripper))
                    
            #         master_delta_pose_xyz = (master_delta[:3] / 1e6) * 2
            #         master_delta_pose_quat = master_delta[3:6] / 1000
                    
            #         return master_delta_pose_xyz, master_delta_pose_quat, master_gripper
            #     else:
            #         return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.05
                    
        elif control_mode == "neural":
            # 神经网络控制模式
            camera_data = get_rgbd(self.cameras)
            camera_images = [image for image, _, _ in camera_data]
            return self.neural_controller.get_action(current_tcp_pos_xyz, current_tcp_quat, camera_images)
            
        elif control_mode == "mode3":
            # 模式3
            camera_data = get_rgbd(self.cameras)
            camera_images = [image for image, _, _ in camera_data]
            return self.mode3_controller.get_action(current_tcp_pos_xyz, current_tcp_quat, camera_images)
            
        else:
            self.logger.warn(f"未知控制模式: {control_mode}，使用mode3")
            camera_data = get_rgbd(self.cameras)
            camera_images = [image for image, _, _ in camera_data]
            return self.mode3_controller.get_action(current_tcp_pos_xyz, current_tcp_quat, camera_images)

    def start_get_replay(self, ):
        try:
            # 如果使用远程通信，启动远程通信服务
            if self.use_remote and self.remote_comm:
                self.remote_comm.start_communication()
                self.logger.info("远程通信服务已启动，等待从臂连接...")
            
            # 打印模式说明
            self.logger.info("="*60)
            self.logger.info("🎮 多模式控制系统已启动 (带超时检测)")
            self.logger.info("默认模式: mode3")
            self.logger.info("客户端超时: 3秒")
            self.logger.info("支持的控制模式:")
            self.logger.info("  - remote: 远程控制模式 (Piper遥控)")
            self.logger.info("  - neural: 神经网络模式 (AI控制)")
            self.logger.info("  - mode3: 模式3 (默认模式)")
            self.logger.info("="*60)
            
            status_log_counter = 0  # 用于定期输出状态信息
            
            while True:
                robot_states = self.robot.states()
                gripper_states = self.gripper.states()

                # Record Traj
                self.add_state(robot_states=robot_states, gripper_states=gripper_states)

                current_tcp_pose = robot_states.tcp_pose
                current_tcp_pos_xyz = np.array(current_tcp_pose[:3])
                current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])

                # 获取远程通信数据（包含控制模式信息）
                pose_data = None
                control_mode = "mode3"  # 默认模式改为mode3
                
                if self.use_remote and self.remote_comm:
                    pose_data = self.remote_comm.get_latest_pose_data()
                    
                    # 更新客户端超时监控
                    self.timeout_monitor.update_client_data(pose_data)
                    
                    # 检查客户端是否超时
                    if self.timeout_monitor.is_client_timeout():
                        control_mode = "mode3"  # 超时时切换到mode3
                    elif pose_data is not None:
                        control_mode = pose_data.get('control_mode', "mode3")
                
                # 更新共享的主机械臂状态变量
                with self.master_state_shared['lock']:
                    self.master_state_shared['current_tcp_pos_xyz'] = current_tcp_pos_xyz.copy()
                    self.master_state_shared['current_tcp_quat'] = current_tcp_quat
                    self.master_state_shared['control_mode'] = control_mode

                # 根据控制模式获取动作指令
                    end_effector_pos,  end_effector_ori, leader_gripper_pos = self.get_action_from_mode(
                    control_mode, current_tcp_pos_xyz, current_tcp_quat, pose_data
                )

                # # 处理姿态变换
                # roll = math.radians(wrap_angle_deg(master_delta_pose_quat[0]))
                # pitch = -math.radians(wrap_angle_deg(master_delta_pose_quat[1]))
                # yaw = -math.radians(wrap_angle_deg(master_delta_pose_quat[2]))

                # q = euler2quat(roll, pitch, yaw, axes='sxyz')  # (w, x, y, z)
                # offset_quat = quaternion.quaternion(*q)

                # _final_pose = current_tcp_pos_xyz + master_delta_pose_xyz
                # _final_quat = current_tcp_quat #* offset_quat
                
                
                self.robot.SendCartesianMotionForce([*end_effector_pos, end_effector_ori.w, end_effector_ori.x, end_effector_ori.y, end_effector_ori.z], 
                                            [0.0] * 6, max_linear_vel = 0.04, max_angular_vel = 0.3,)
            
                self.gripper.Move(leader_gripper_pos, 0.2, 20)
                
                # 发送压缩后的图像数据到从臂（仅在远程控制模式下且客户端连接时）
                if self.use_remote and self.remote_comm and control_mode == "remote" and not self.timeout_monitor.is_client_timeout():
                    camera_data = get_rgbd(self.cameras)
                    if len(camera_data) == self.num_cameras:
                        images_data = {}
                        for i, (image, _, _) in enumerate(camera_data):
                            cam_key = f'cam{i+1}'
                            images_data[cam_key] = image
                        self.remote_comm.send_images(images_data)
                
                # 定期输出状态信息
                status_log_counter += 1
                if status_log_counter % 100 == 0:  # 每100个循环输出一次状态
                    self.logger.info(f"当前模式: {control_mode} | {self.timeout_monitor.get_status_info()}")
                
                time.sleep(0.03)

        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 停止子机械臂线程
            self.stop_sub_arms()
            
            # save traj
            # self.save_trajectory()
            # print("save traj successfull")

            # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
            # robot.ExecutePlan("PLAN-Home")
            self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
            self.robot.SendCartesianMotionForce(self.flexiv_init_pose, [0.0] * 6, max_linear_vel = 0.04, max_angular_vel = 0.3,)
            while self.robot.busy():
                time.sleep(0.02)

            # Force control, if available (sensed force is not zero)
            if abs(self.gripper.states().force) > sys.float_info.epsilon:
                self.logger.info("Gripper running zero force control")
                self.gripper.Grasp(0)
                # Exit after 10 seconds
                time.sleep(3)

            # Finished
            self.gripper.Stop()
            
            # 停止远程通信
            if self.use_remote and self.remote_comm:
                self.remote_comm.stop_communication()

    def save_trajectory(self, ):
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

            # Save metadata
            hf.attrs['instruction'] = self.task
            hf.attrs['num_frames'] = len(self.timestamps)
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hf.attrs['num_cameras'] = self.camera_num

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
            
        self.logger.info(f"Task: {self.task}, Frames: {len(self.timestamps)}, Saved to: {self.output_file}")


if __name__ == "__main__":
    # 配置参数 - 直接在这里修改，不需要命令行参数
    use_remote = True  # 设置为True使用远程通信，False使用本地从臂
    remote_ip = "192.168.1.100"  # 从臂电脑IP，如果使用远程通信需要设置
    image_quality = 50  # JPEG压缩质量 (1-100)，数值越小压缩率越高，但图像质量越低
    client_timeout = 3.0  # 客户端超时时间（秒）
    
    task_config = {
        "task_name": "multimode_control_timeout",
        "output_file": f"/home/pjlab/work/flexiv/zty_data/{time.time()}.h5",
        "camera_config": {
            "camera_num": 2,
            "camera_info": {
                "242322073804": "above",
                "239722070506": "gripper"
            }
        }
    }
    
    print("=== 多模式主臂程序启动 (带超时检测) ===")
    print(f"使用远程通信: {use_remote}")
    print(f"默认模式: mode3")
    print(f"客户端超时: {client_timeout}秒")
    if use_remote:
        print(f"从臂IP地址: {remote_ip}")
        print(f"图像压缩质量: {image_quality}")
    print("支持多种控制模式切换，带客户端超时保护")
    print("按Ctrl+C停止程序")
    print("")
                    
    Recorder = Zty_TrajectoryRecorder(
        task_config=task_config, 
        use_remote=use_remote, 
        remote_ip=remote_ip,
        image_quality=image_quality,
        client_timeout=client_timeout
    )
    Recorder.start_get_replay()
