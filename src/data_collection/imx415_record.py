import os
import cv2
import numpy as np
import time
import argparse
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class IMX415CameraConfig:
    """Configuration parameters for the IMX415 camera setup"""
    real_time_view: bool = False
    image_size: Tuple[int, int] = (1920, 1080)  # IMX415 常用分辨率
    fps: int = 30
    save_path: str = './imx415/images'
    save_freq: int = 10
    device_paths: List[str] = None  # 摄像头设备路径列表，如 ['/dev/video0', '/dev/video1']
    fourcc: str = 'MJPG'  # 视频编码格式
    buffer_size: int = 1  # 缓冲区大小


class IMX415Module:
    def __init__(self, config: IMX415CameraConfig = None):
        if config is None:
            config = IMX415CameraConfig()
        self.config = config
        self.captures = []
        self.device_count = 0
        
        # 如果没有指定设备路径，自动检测可用摄像头
        if config.device_paths is None:
            self.config.device_paths = self._detect_cameras()
        
        if not self.config.device_paths:
            raise RuntimeError("No IMX415 cameras detected")
        
        # 初始化所有摄像头
        self._setup_cameras()
        
        # 创建实时预览窗口
        if config.real_time_view:
            for i in range(len(self.config.device_paths)):
                device_name = os.path.basename(self.config.device_paths[i])
                cv2.namedWindow(f"IMX415_{device_name}", cv2.WINDOW_AUTOSIZE)
    
    def _detect_cameras(self) -> List[str]:
        """自动检测可用的摄像头设备"""
        available_cameras = []
        # 检测前8个设备号
        for i in range(8):
            device_path = f"/dev/video{i}"
            # 检查设备文件是否存在
            if os.path.exists(device_path):
                cap = cv2.VideoCapture(device_path)
                if cap.isOpened():
                    # 尝试读取一帧来确认摄像头可用
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(device_path)
                        print(f"检测到摄像头设备: {device_path}")
                cap.release()
        return available_cameras
    
    def _setup_cameras(self):
        """设置所有摄像头"""
        for device_path in self.config.device_paths:
            cap = cv2.VideoCapture(device_path)
            
            if not cap.isOpened():
                print(f"警告: 无法打开摄像头设备 {device_path}")
                continue
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.image_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.image_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 设置编码格式
            fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            # 设置缓冲区大小以减少延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # 验证设置是否成功
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"摄像头 {device_path} 设置:")
            print(f"  分辨率: {actual_width}x{actual_height}")
            print(f"  帧率: {actual_fps}")
            
            self.captures.append(cap)
        
        self.device_count = len(self.captures)
        if self.device_count == 0:
            raise RuntimeError("没有成功初始化任何摄像头")
    
    def get_camera_info(self, cap_index: int = 0) -> dict:
        """获取摄像头信息"""
        if cap_index >= len(self.captures):
            return {}
        
        cap = self.captures[cap_index]
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': cap.get(cv2.CAP_PROP_SATURATION),
            'hue': cap.get(cv2.CAP_PROP_HUE),
            'gain': cap.get(cv2.CAP_PROP_GAIN),
            'exposure': cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        return info
    
    def get_data(self) -> List[np.ndarray]:
        """从所有摄像头获取图像数据"""
        images = []
        
        for i, cap in enumerate(self.captures):
            ret, frame = cap.read()
            if not ret:
                print(f"警告: 无法从摄像头 {self.config.device_paths[i]} 读取帧")
                # 返回空图像作为占位符
                empty_frame = np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)
                images.append(empty_frame)
            else:
                images.append(frame)
        
        return images
    
    def get_single_frame(self, cam_index: int = 0) -> Optional[np.ndarray]:
        """从指定摄像头获取单帧图像"""
        if cam_index >= len(self.captures):
            print(f"错误: 摄像头索引 {cam_index} 超出范围")
            return None
        
        ret, frame = self.captures[cam_index].read()
        return frame if ret else None
    
    def show_real_time_view(self):
        """显示实时预览"""
        if not self.config.real_time_view:
            return
        
        images = self.get_data()
        for i, image in enumerate(images):
            if image is not None:
                cv2.imshow(f"IMX415_Camera_{i+1}", image)
    
    def cleanup(self):
        """清理资源"""
        for cap in self.captures:
            try:
                cap.release()
            except Exception as e:
                print(f"释放摄像头时出错: {e}")
        
        if self.config.real_time_view:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"关闭窗口时出错: {e}")


def save_image_sequences(imx415_module: IMX415Module, config: IMX415CameraConfig):
    """保存图像序列"""
    os.makedirs(config.save_path, exist_ok=True)
    frame_count = 0
    time_interval = 1.0 / config.save_freq
    
    try:
        while True:
            images = imx415_module.get_data()
            
            for cam_idx, image in enumerate(images):
                if image is not None:
                    # 保存为numpy数组
                    np.save(os.path.join(config.save_path, f'image_cam{cam_idx}_{frame_count:06d}.npy'), image)
                    # 保存为JPEG图像
                    cv2.imwrite(os.path.join(config.save_path, f'image_cam{cam_idx}_{frame_count:06d}.jpg'), image)
            
            # 显示实时预览
            if config.real_time_view:
                imx415_module.show_real_time_view()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            print(f'保存帧: {frame_count}')
            time.sleep(time_interval)
    
    except KeyboardInterrupt:
        print("用户中断保存")
    finally:
        imx415_module.cleanup()


def get_images(imx415_module: IMX415Module) -> List[np.ndarray]:
    """获取所有摄像头的图像"""
    return imx415_module.get_data()


def parse_args() -> IMX415CameraConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IMX415 Camera Data Capture")
    parser.add_argument('--real-time-view', action='store_true', help='启用实时预览')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度')
    parser.add_argument('--height', type=int, default=1080, help='图像高度')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    parser.add_argument('--save-path', type=str, default='./imx415/images', 
                       help='图像保存路径')
    parser.add_argument('--save-freq', type=int, default=10, help='保存频率 (Hz)')
    parser.add_argument('--device-paths', type=str, nargs='+', 
                       help='摄像头设备路径列表，如 /dev/video0 /dev/video1')
    parser.add_argument('--fourcc', type=str, default='MJPG', 
                       help='视频编码格式 (MJPG, YUYV, etc.)')
    parser.add_argument('--buffer-size', type=int, default=1, help='缓冲区大小')
    
    args = parser.parse_args()
    return IMX415CameraConfig(
        real_time_view=args.real_time_view,
        image_size=(args.width, args.height),
        fps=args.fps,
        save_path=args.save_path,
        save_freq=args.save_freq,
        device_paths=args.device_paths,
        fourcc=args.fourcc,
        buffer_size=args.buffer_size
    )


if __name__ == '__main__':
    config = parse_args()
    
    try:
        cameras = IMX415Module(config)
        print(f"成功初始化 {cameras.device_count} 个摄像头")
        
        # 打印摄像头信息
        for i in range(cameras.device_count):
            info = cameras.get_camera_info(i)
            print(f"\n摄像头 {i} 信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # 获取并显示图像
        images = get_images(cameras)
        for i, image in enumerate(images):
            if image is not None:
                cv2.imshow(f'IMX415_Camera_{i+1}', image)
        
        print("按任意键继续，或取消注释下面的代码来保存图像序列...")
        cv2.waitKey(0)
        
        # 取消注释以保存图像序列
        # save_image_sequences(cameras, config)
    
    except Exception as e:
        print(f"错误: {e}")
    finally:
        try:
            cameras.cleanup()
        except:
            pass
        cv2.destroyAllWindows()

# 使用示例:
# python imx415_record.py --real-time-view --width 1920 --height 1080 --fps 30 --save-path ./data/imx415
# python imx415_record.py --device-paths /dev/video0 /dev/video2 --real-time-view  # 指定使用特定设备
# python imx415_record.py --device-paths /dev/video0 --real-time-view  # 使用单个设备
