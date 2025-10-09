#!/usr/bin/env python3
"""
IMX415摄像头测试脚本
用于调试摄像头访问问题
"""

import os
import cv2
import subprocess
import sys

def check_device(device_path):
    """检查指定设备的状态"""
    print(f"=== 检查设备: {device_path} ===")
    
    # 1. 检查设备文件是否存在
    if not os.path.exists(device_path):
        print(f"❌ 设备文件不存在: {device_path}")
        return False
    else:
        print(f"✅ 设备文件存在: {device_path}")
    
    # 2. 检查设备权限
    try:
        stat_info = os.stat(device_path)
        permissions = oct(stat_info.st_mode)[-3:]
        print(f"📁 设备权限: {permissions}")
        
        # 检查是否可读可写
        readable = os.access(device_path, os.R_OK)
        writable = os.access(device_path, os.W_OK)
        print(f"📖 可读: {readable}, 可写: {writable}")
        
        if not (readable and writable):
            print("⚠️  权限不足，建议运行: sudo chmod 666 " + device_path)
            
    except Exception as e:
        print(f"❌ 检查权限时出错: {e}")
    
    # 3. 尝试使用OpenCV打开设备
    print("\n🔍 测试OpenCV访问...")
    
    # 提取设备号
    try:
        device_num = int(device_path.split('video')[1])
        print(f"设备号: {device_num}")
    except:
        print("❌ 无法解析设备号")
        return False
    
    # 测试不同的打开方式
    test_methods = [
        ("设备路径", device_path),
        ("设备号", device_num),
    ]
    
    for method_name, device_id in test_methods:
        print(f"\n尝试方法: {method_name} ({device_id})")
        try:
            cap = cv2.VideoCapture(device_id)
            
            if cap.isOpened():
                print("✅ 设备打开成功")
                
                # 获取设备属性
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   分辨率: {width}x{height}")
                print(f"   帧率: {fps}")
                
                # 尝试读取一帧
                print("   尝试读取帧...")
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ 成功读取帧，形状: {frame.shape}")
                    
                    # 保存测试图像
                    cv2.imwrite(f"test_frame_{device_num}.jpg", frame)
                    print(f"💾 测试图像已保存: test_frame_{device_num}.jpg")
                    
                    cap.release()
                    return True
                else:
                    print(f"❌ 无法读取帧 (ret={ret})")
            else:
                print("❌ 无法打开设备")
                
            cap.release()
            
        except Exception as e:
            print(f"❌ 测试时出错: {e}")
    
    return False

def check_v4l2_info(device_path):
    """使用v4l2-ctl检查设备信息"""
    print(f"\n=== V4L2设备信息: {device_path} ===")
    
    try:
        # 检查设备能力
        result = subprocess.run(['v4l2-ctl', '--device', device_path, '--all'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"v4l2-ctl 错误: {result.stderr}")
            
    except FileNotFoundError:
        print("v4l2-ctl 未找到，安装命令: sudo apt-get install v4l-utils")
    except subprocess.TimeoutExpired:
        print("v4l2-ctl 超时")
    except Exception as e:
        print(f"运行 v4l2-ctl 时出错: {e}")

def suggest_solutions(device_path):
    """提供解决方案建议"""
    print(f"\n=== 解决方案建议 ===")
    
    print("1. 检查权限:")
    print(f"   sudo chmod 666 {device_path}")
    print("   或者将用户添加到video组: sudo usermod -a -G video $USER")
    
    print("\n2. 检查设备是否被占用:")
    print(f"   sudo fuser {device_path}")
    print("   如果有进程占用，使用: sudo fuser -k " + device_path)
    
    print("\n3. 重启设备:")
    print("   sudo modprobe -r uvcvideo && sudo modprobe uvcvideo")
    
    print("\n4. 检查系统日志:")
    print("   dmesg | tail -20")
    print("   dmesg | grep video")
    
    print("\n5. 手动测试:")
    print(f"   ffmpeg -f v4l2 -i {device_path} -frames:v 1 test.jpg")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        device_path = sys.argv[1]
    else:
        device_path = "/dev/video12"  # 默认设备
    
    print("IMX415摄像头测试工具")
    print("=" * 50)
    
    # 检查设备
    success = check_device(device_path)
    
    # 检查V4L2信息
    check_v4l2_info(device_path)
    
    # 提供建议
    if not success:
        suggest_solutions(device_path)
    else:
        print(f"\n✅ 设备 {device_path} 工作正常！")

if __name__ == "__main__":
    main()
