import os
import cv2
import numpy as np
import time
import argparse
import fcntl
import mmap
import select
import struct
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

# V4L2 constants
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xc0405602
VIDIOC_S_FMT = 0xc0d05605
VIDIOC_G_FMT = 0xc0d05604
VIDIOC_REQBUFS = 0xc0145608
VIDIOC_QUERYBUF = 0xc0445609
VIDIOC_QBUF = 0xc044560f
VIDIOC_DQBUF = 0xc0445611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0

# 像素格式
V4L2_PIX_FMT_YUYV = 0x56595559  # YUYV 4:2:2
V4L2_PIX_FMT_MJPEG = 0x47504A4D  # MJPG


class V4L2Camera:
    """基于V4L2的高性能摄像头类"""
    
    def __init__(self, device_path: str, width: int = 1280, height: int = 720, 
                 pixel_format: str = 'YUYV', fps: int = 30, buffer_count: int = 4):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = buffer_count
        self.fd = None
        self.buffers = []
        self.streaming = False
        
        # 设置像素格式
        if pixel_format == 'MJPEG':
            self.pixel_format = V4L2_PIX_FMT_MJPEG
        else:
            self.pixel_format = V4L2_PIX_FMT_YUYV
            
        self.pixel_format_name = pixel_format
        
        self._open_device()
        self._setup_format()
        self._setup_buffers()
        
    def _open_device(self):
        """打开V4L2设备"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
            print(f"✅ V4L2设备已打开: {self.device_path}")
        except OSError as e:
            raise RuntimeError(f"无法打开V4L2设备 {self.device_path}: {e}")
    
    def _setup_format(self):
        """设置视频格式"""
        # 构造格式结构体
        fmt = struct.pack(
            '=LLLLLLLLLLLLLL',
            V4L2_BUF_TYPE_VIDEO_CAPTURE,  # type
            self.width,                    # width
            self.height,                   # height
            self.pixel_format,             # pixelformat
            V4L2_FIELD_ANY,               # field
            self.width * 2 if self.pixel_format == V4L2_PIX_FMT_YUYV else 0,  # bytesperline
            self.width * self.height * 2 if self.pixel_format == V4L2_PIX_FMT_YUYV else 0,  # sizeimage
            0, 0, 0, 0, 0, 0, 0           # 剩余字段
        )
        
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
            print(f"✅ 设置格式: {self.width}x{self.height} {self.pixel_format_name}")
        except OSError as e:
            raise RuntimeError(f"设置视频格式失败: {e}")
    
    def _setup_buffers(self):
        """设置缓冲区"""
        # 请求缓冲区
        reqbuf = struct.pack('=LLLL', self.buffer_count, V4L2_BUF_TYPE_VIDEO_CAPTURE, 
                            V4L2_MEMORY_MMAP, 0)
        
        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbuf)
            print(f"✅ 请求了 {self.buffer_count} 个缓冲区")
        except OSError as e:
            raise RuntimeError(f"请求缓冲区失败: {e}")
        
        # 查询并映射每个缓冲区
        for i in range(self.buffer_count):
            querybuf = struct.pack('=LLLLLLLLLLLL', i, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            try:
                result = fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, querybuf)
                # 解析返回的缓冲区信息
                unpacked = struct.unpack('=LLLLLLLLLLLL', result)
                offset = unpacked[9]  # m.offset
                length = unpacked[4]  # length
                
                # 内存映射
                buffer_mem = mmap.mmap(self.fd, length, 
                                     mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                                     offset=offset)
                
                self.buffers.append({
                    'index': i,
                    'length': length,
                    'offset': offset,
                    'mmap': buffer_mem
                })
                
            except OSError as e:
                raise RuntimeError(f"查询/映射缓冲区 {i} 失败: {e}")
        
        print(f"✅ 映射了 {len(self.buffers)} 个缓冲区")
    
    def start_streaming(self):
        """开始流传输"""
        if self.streaming:
            return
            
        # 将所有缓冲区加入队列
        for buffer_info in self.buffers:
            qbuf = struct.pack('=LLLLLLLLLLLL', buffer_info['index'], 
                              V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            try:
                fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)
            except OSError as e:
                raise RuntimeError(f"缓冲区入队失败: {e}")
        
        # 启动流
        buf_type = struct.pack('=L', V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type)
            self.streaming = True
            print("✅ 流传输已启动")
        except OSError as e:
            raise RuntimeError(f"启动流失败: {e}")
    
    def stop_streaming(self):
        """停止流传输"""
        if not self.streaming:
            return
            
        buf_type = struct.pack('=L', V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type)
            self.streaming = False
            print("✅ 流传输已停止")
        except OSError as e:
            print(f"停止流失败: {e}")
    
    def read_frame(self, timeout=1.0):
        """读取一帧数据"""
        if not self.streaming:
            self.start_streaming()
        
        # 使用select等待数据就绪
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        
        # 出队一个缓冲区
        dqbuf = struct.pack('=LLLLLLLLLLLL', 0, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        try:
            result = fcntl.ioctl(self.fd, VIDIOC_DQBUF, dqbuf)
            unpacked = struct.unpack('=LLLLLLLLLLLL', result)
            index = unpacked[0]
            bytesused = unpacked[4]
            
            # 读取数据
            buffer_info = self.buffers[index]
            buffer_info['mmap'].seek(0)
            data = buffer_info['mmap'].read(bytesused)
            
            # 重新入队
            qbuf = struct.pack('=LLLLLLLLLLLL', index, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)
            
            # 转换数据为numpy数组
            if self.pixel_format == V4L2_PIX_FMT_MJPEG:
                # MJPEG需要解码
                frame = self._decode_mjpeg(data)
            else:
                # YUYV转BGR
                frame = self._yuyv_to_bgr(data)
            
            return frame
            
        except OSError as e:
            print(f"读取帧失败: {e}")
            return None
    
    def _decode_mjpeg(self, data):
        """解码MJPEG数据"""
        try:
            # 使用OpenCV解码MJPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"MJPEG解码失败: {e}")
            return None
    
    def _yuyv_to_bgr(self, data):
        """将YUYV格式转换为BGR"""
        try:
            # YUYV是4:2:2格式，每2个像素占4字节
            yuyv = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width * 2))
            
            # 转换为BGR
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            return bgr
        except Exception as e:
            print(f"YUYV转换失败: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        self.stop_streaming()
        
        # 解除内存映射
        for buffer_info in self.buffers:
            if buffer_info['mmap']:
                buffer_info['mmap'].close()
        
        # 关闭设备
        if self.fd is not None:
            os.close(self.fd)
            print("✅ V4L2设备已关闭")


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
        self.cameras = []  # 改为存储V4L2Camera对象
        self.device_count = 0
        
        # 如果没有指定设备路径，使用默认路径
        if config.device_paths is None:
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
        """设置所有V4L2摄像头"""
        for device_path in self.config.device_paths:
            print(f"初始化V4L2摄像头: {device_path}")
            
            try:
                # 创建V4L2Camera实例
                camera = V4L2Camera(
                    device_path=device_path,
                    width=self.config.image_size[0],
                    height=self.config.image_size[1],
                    pixel_format=self.config.fourcc,
                    fps=self.config.fps,
                    buffer_count=self.config.buffer_size
                )
                
                # 测试读取一帧
                test_frame = camera.read_frame(timeout=2.0)
                if test_frame is not None:
                    print(f"✅ V4L2摄像头初始化成功: {device_path}")
                    print(f"   实际分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    print(f"   像素格式: {self.config.fourcc}")
                    self.cameras.append(camera)
                else:
                    print(f"❌ 无法从V4L2设备读取帧: {device_path}")
                    camera.cleanup()
                    
            except Exception as e:
                print(f"❌ V4L2摄像头初始化失败 {device_path}: {e}")
                continue
        
        self.device_count = len(self.cameras)
        if self.device_count == 0:
            error_msg = f"无法初始化任何V4L2摄像头设备: {self.config.device_paths}\n"
            error_msg += "请确认:\n"
            error_msg += "1. 设备路径是否正确\n"
            error_msg += "2. 摄像头权限是否足够: sudo chmod 666 /dev/video*\n" 
            error_msg += "3. 设备是否被其他程序占用\n"
            error_msg += "4. 设备是否支持指定的分辨率和格式"
            raise RuntimeError(error_msg)
    
    def get_camera_info(self, cap_index: int = 0) -> dict:
        """获取摄像头信息"""
        if cap_index >= len(self.cameras):
            return {}
        
        camera = self.cameras[cap_index]
        info = {
            'width': camera.width,
            'height': camera.height,
            'fps': camera.fps,
            'pixel_format': camera.pixel_format_name,
            'buffer_count': camera.buffer_count,
            'device_path': camera.device_path,
            'streaming': camera.streaming
        }
        return info
    
    def get_data(self) -> List[np.ndarray]:
        """从所有摄像头获取图像数据"""
        images = []
        
        for i, camera in enumerate(self.cameras):
            frame = camera.read_frame(timeout=0.1)  # 100ms超时
            if frame is None:
                print(f"警告: 无法从V4L2摄像头 {self.config.device_paths[i]} 读取帧")
                # 返回空图像作为占位符
                empty_frame = np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)
                images.append(empty_frame)
            else:
                images.append(frame)
        
        return images
    
    def get_single_frame(self, cam_index: int = 0) -> Optional[np.ndarray]:
        """从指定摄像头获取单帧图像"""
        if cam_index >= len(self.cameras):
            print(f"错误: 摄像头索引 {cam_index} 超出范围")
            return None
        
        return self.cameras[cam_index].read_frame(timeout=1.0)
    
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
        for camera in self.cameras:
            try:
                camera.cleanup()
            except Exception as e:
                print(f"清理V4L2摄像头时出错: {e}")
        
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
                       help='像素格式 (YUYV, MJPEG)')
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
def test_v4l2_performance():
    """测试V4L2不同配置的性能"""
    test_configs = [
        {'size': (640, 480), 'format': 'YUYV', 'buffers': 4},
        {'size': (640, 480), 'format': 'MJPEG', 'buffers': 4},
        {'size': (1280, 720), 'format': 'YUYV', 'buffers': 4},
        {'size': (1280, 720), 'format': 'MJPEG', 'buffers': 4},
        {'size': (1920, 1080), 'format': 'YUYV', 'buffers': 4},
        {'size': (1920, 1080), 'format': 'MJPEG', 'buffers': 4},
    ]
    
    print("=== V4L2性能测试 ===")
    
    for config in test_configs:
        print(f"\n📊 测试: {config['size']} {config['format']}")
        
        cam_config = IMX415CameraConfig(
            image_size=config['size'],
            fourcc=config['format'],
            buffer_size=config['buffers'],
            device_paths=['/dev/video12']
        )
        
        try:
            camera = IMX415Module(cam_config)
            
            # 预热
            for _ in range(3):
                camera.get_data()
            
            # 测试50帧的性能
            frame_count = 50
            start_time = time.time()
            
            successful_frames = 0
            for i in range(frame_count):
                images = camera.get_data()
                if images and len(images) > 0 and images[0] is not None:
                    successful_frames += 1
            
            total_time = time.time() - start_time
            avg_fps = successful_frames / total_time
            print(f"   ✅ 实际帧率: {avg_fps:.2f} fps")
            print(f"   📈 成功率: {successful_frames}/{frame_count} ({100*successful_frames/frame_count:.1f}%)")
            print(f"   ⏱️  总时间: {total_time:.2f}s")
            
            camera.cleanup()
            
        except Exception as e:
            print(f"   ❌ 配置失败: {e}")

def compare_opencv_vs_v4l2():
    """比较OpenCV和V4L2的性能"""
    print("\n=== OpenCV vs V4L2 性能比较 ===")
    
    # 测试OpenCV
    print("\n🔵 OpenCV VideoCapture:")
    try:
        cap = cv2.VideoCapture('/dev/video12')
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 预热
            for _ in range(5):
                cap.read()
            
            start_time = time.time()
            successful = 0
            for _ in range(30):
                ret, frame = cap.read()
                if ret:
                    successful += 1
            
            total_time = time.time() - start_time
            fps = successful / total_time
            print(f"   帧率: {fps:.2f} fps, 成功率: {successful}/30")
            cap.release()
        else:
            print("   无法打开OpenCV设备")
    except Exception as e:
        print(f"   OpenCV测试失败: {e}")
    
    # 测试V4L2
    print("\n🟢 V4L2直接访问:")
    try:
        v4l2_cam = V4L2Camera('/dev/video12', 1280, 720, 'YUYV', buffer_count=4)
        
        # 预热
        for _ in range(5):
            v4l2_cam.read_frame()
        
        start_time = time.time()
        successful = 0
        for _ in range(30):
            frame = v4l2_cam.read_frame(timeout=0.1)
            if frame is not None:
                successful += 1
        
        total_time = time.time() - start_time
        fps = successful / total_time
        print(f"   帧率: {fps:.2f} fps, 成功率: {successful}/30")
        v4l2_cam.cleanup()
    except Exception as e:
        print(f"   V4L2测试失败: {e}")

if __name__ == '__main__':
    # 性能测试
    test_v4l2_performance()
    compare_opencv_vs_v4l2()
# V4L2高性能IMX415摄像头模块使用示例:

# 1. 性能测试:
# python imx415_record.py  # 运行性能对比测试

# 2. 基本使用 (默认使用 /dev/video12):
# python imx415_record.py --real-time-view

# 3. 指定设备和格式:
# python imx415_record.py --device-paths /dev/video12 --fourcc YUYV --real-time-view
# python imx415_record.py --device-paths /dev/video12 --fourcc MJPEG --real-time-view

# 4. 高分辨率设置:
# python imx415_record.py --device-paths /dev/video12 --width 1920 --height 1080 --fourcc MJPEG --real-time-view

# 5. 多摄像头 (如果有):
# python imx415_record.py --device-paths /dev/video12 /dev/video13 --real-time-view

# 6. 优化设置 (减少延迟):
# python imx415_record.py --device-paths /dev/video12 --fourcc YUYV --buffer-size 2 --real-time-view

# 注意事项:
# - YUYV格式CPU占用更高但延迟更低
# - MJPEG格式CPU占用更低但可能有轻微延迟
# - 较小的buffer_size可以减少延迟
# - 确保设备权限: sudo chmod 666 /dev/video*
