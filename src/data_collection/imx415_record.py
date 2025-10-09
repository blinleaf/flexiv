import os
import cv2
import numpy as np
import time
import argparse
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class IMX415CameraConfig:
    """Configuration parameters for the IMX415 camera setup"""
    real_time_view: bool = False
    image_size: Tuple[int, int] = (1280, 720)  # IMX415 常用分辨率
    fps: int = 30
    save_path: str = './imx415/images'
    save_freq: int = 30
    device_paths: Optional[List[str]] = field(default=None)  # 摄像头设备路径列表，如 ['/dev/video0', '/dev/video1']
    fourcc: str = 'MJPG'  # 视频编码格式
    buffer_size: int = 5  # 缓冲区大小


class IMX415Module:
    def __init__(self, config: IMX415CameraConfig = None):
        if config is None:
            config = IMX415CameraConfig()
        self.config = config
        self.captures = []
        self.device_count = 0
        
        # 如果没有指定设备路径，使用默认路径
        if config.device_paths is None:
            # self.config.device_paths = ['/dev/video12', '/dev/video13']  # 默认路径
            # print("未指定设备路径，使用默认设备 /dev/video0")
            # self.config.device_paths = ["/dev/video0"]
            self.config.device_paths = ['/dev/video12']
        
        print(f"使用摄像头设备: {self.config.device_paths}")
        
        # 初始化所有摄像头
        self._setup_cameras()
        
        # 创建实时预览窗口
        if config.real_time_view:
            for i in range(len(self.config.device_paths)):
                device_name = os.path.basename(self.config.device_paths[i])
                cv2.namedWindow(f"IMX415_{device_name}", cv2.WINDOW_AUTOSIZE)
    
    
    def _setup_cameras(self):
        """设置所有摄像头"""
        for device_path in self.config.device_paths:
            print(f"尝试打开设备: {device_path}")
            
            # 尝试两种方式：设备路径和设备号
            cap = None
            success = False
            
            # 方式1: 直接使用设备路径
            try:
                cap = cv2.VideoCapture(device_path)
                if cap.isOpened():
                    # 尝试读取一帧来验证
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ 使用设备路径成功: {device_path}")
                        success = True
                    else:
                        cap.release()
                        cap = None
            except:
                if cap:
                    cap.release()
                cap = None
            
            # 方式2: 使用设备号
            if not success:
                try:
                    device_num = int(device_path.split('video')[1])
                    print(f"尝试使用设备号: {device_num}")
                    cap = cv2.VideoCapture(device_num)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✅ 使用设备号成功: {device_num}")
                            success = True
                        else:
                            cap.release()
                            cap = None
                except:
                    if cap:
                        cap.release()
                    cap = None
            
            if not success:
                print(f"❌ 无法打开摄像头设备 {device_path}")
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
            error_msg = f"无法初始化摄像头设备: {self.config.device_paths}\n"
            error_msg += "请确认:\n"
            error_msg += "1. 设备路径是否正确\n"
            error_msg += "2. 摄像头权限是否足够: sudo chmod 666 /dev/video*\n" 
            error_msg += "3. 设备是否被其他程序占用"
            raise RuntimeError(error_msg)
    
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
    parser.add_argument('--save-freq', type=int, default=30, help='保存频率 (Hz)')
    parser.add_argument('--device-paths', type=str, nargs='+', 
                       help='摄像头设备路径列表，如 /dev/video0 /dev/video1')
    parser.add_argument('--fourcc', type=str, default='YUYV', 
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
        
        while True:
            start_time = time.time()
            images = get_images(cameras)
            print('len(images): ', len(images))
            # for i, image in enumerate(images):
            #     if image is not None:
            #         cv2.imshow(f'IMX415_Camera_{i+1}', image)
            
            # print("按任意键继续，或取消注释下面的代码来保存图像序列...")
            print('len: ', time.time() - start_time)
            # cv2.waitKey(0)
        
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
def test_camera_performance():
    """测试不同设置的摄像头性能"""
    test_configs = [
        {'size': (640, 480), 'fourcc': 'MJPG', 'buffer': 4},
        {'size': (1280, 720), 'fourcc': 'MJPG', 'buffer': 4},
        {'size': (1920, 1080), 'fourcc': 'MJPG', 'buffer': 4},
        {'size': (3840, 2160), 'fourcc': 'MJPG', 'buffer': 4},
    ]
    
    for config in test_configs:
        print(f"\n测试配置: {config['size']}, {config['fourcc']}")
        
        cam_config = IMX415CameraConfig(
            image_size=config['size'],
            fourcc=config['fourcc'],
            buffer_size=config['buffer'],
            device_paths=['/dev/video12']
        )
        
        try:
            camera = IMX415Module(cam_config)
            
            # 测试10帧的性能
            start_time = time.time()
            frame_count = 0
            for i in range(10):
                images = camera.get_data()
                frame_count += 1
            
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"平均帧率: {avg_fps:.2f} Hz")
            
            camera.cleanup()
            
        except Exception as e:
            print(f"配置失败: {e}")

# 在main函数中调用测试
if __name__ == '__main__':
    # 先测试性能
    test_camera_performance()
# 使用示例:
# 使用默认设备 /dev/video0:
# python imx415_record.py --real-time-view

# 指定特定设备:
# python imx415_record.py --device-paths /dev/video0 --real-time-view  
# python imx415_record.py --device-paths /dev/video0 /dev/video2 --real-time-view  # 多个设备

# 完整参数示例:
# python imx415_record.py --device-paths /dev/video0 --real-time-view --width 1920 --height 1080 --fps 30
