#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于YOLOv11姿态估计的摔倒检测系统主程序

此程序用于启动摔倒检测系统的图形界面，支持视频文件、摄像头和静态图片输入，
通过分析人体关键点和角度来检测摔倒事件。
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到Python路径，确保可以正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/fall_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入PyQt5相关模块
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

    logger.info("PyQt5模块导入成功")
except ImportError as e:
    logger.error(f"PyQt5模块导入失败: {str(e)}")
    print("错误: 请安装PyQt5库，使用命令 'pip install PyQt5'")
    sys.exit(1)

# 导入自定义模块
try:
    from ui.main_window import MainWindow

    logger.info("自定义模块导入成功")
except ImportError as e:
    logger.error(f"自定义模块导入失败: {str(e)}")
    print(f"错误: 模块导入失败 - {str(e)}")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于YOLOv11姿态估计的摔倒检测系统')

    # 模型参数
    parser.add_argument('--model', type=str, default='models/yolov11n-pose.pt',
                        help='YOLO模型文件路径')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备，可选cuda:0, cuda或cpu，auto表示自动选择')
    parser.add_argument('--conf-thres', type=float, default=0.4,
                        help='置信度阈值，建议值: 0.35-0.45')
    parser.add_argument('--iou-thres', type=float, default=0.55,
                        help='IOU阈值，建议值: 0.5-0.6')

    # 分析参数
    parser.add_argument('--time-window', type=int, default=8,
                        help='摔倒检测的时间窗口大小，建议值: 5-10')
    
    # 测试和评估参数
    parser.add_argument('--evaluate', action='store_true',
                        help='使用测试集评估模型性能')
    parser.add_argument('--test-dir', type=str, default='datasets/images/test',
                        help='测试图像目录')

    # 界面参数
    parser.add_argument('--fullscreen', action='store_true',
                        help='是否全屏显示')

    return parser.parse_args()


def prepare_model(model_path):
    """准备模型文件，确保模型存在或尝试下载"""
    # 支持多种YOLOv11模型名称格式
    yolo11_models = [
        'yolov11n-pose',
        'yolo11n-pose.pt',
        'yolov11-pose',
        'yolov11-pose.pt',
        'n-pose',
        'yolo11n'
    ]
    
    # 检查模型目录是否存在
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"创建models目录: {models_dir}")
    
    # 首先检查完整路径
    if os.path.exists(model_path):
        logger.info(f"使用指定的模型路径: {model_path}")
        return model_path
    
    # 检查models目录下是否有模型
    if not os.path.isabs(model_path):
        full_model_path = os.path.join(models_dir, model_path)
        if os.path.exists(full_model_path):
            logger.info(f"使用models目录中的模型: {full_model_path}")
            return full_model_path
    
    # 如果模型不存在且是YOLOv11，尝试不同的模型名称格式
    if any('yolov11' in part.lower() for part in [model_path]):
        logger.warning(f"模型文件不存在: {model_path}")
        logger.info("将尝试使用不同格式的YOLOv11模型名称")
        
        # 返回一个标准的YOLOv11模型名称，让YOLO类自动处理下载
        return 'yolov11n-pose'
    
    logger.info("将使用指定的模型名称，让YOLO类自动处理")
    return model_path

def main():
    """主函数"""
    logger.info("=== 基于YOLOv11姿态估计的摔倒检测系统启动 ===")

    # 解析命令行参数
    args = parse_args()
    
    # 自动设备选择
    if args.device.lower() == 'auto':
        try:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            args.device = device
            logger.info(f"自动选择运行设备: {device}")
        except ImportError:
            args.device = 'cpu'
            logger.warning("未安装PyTorch，默认使用CPU模式")
    
    logger.info(f"命令行参数: {args}")
    
    # 检查模型目录和文件
    if not os.path.exists(args.model):
        logger.warning(f"指定的模型文件不存在: {args.model}")
        suggested_model = prepare_model(args.model)
        logger.info(f"将使用模型: {suggested_model}")
        # 更新args中的模型路径
        args.model = suggested_model
    
    # 创建PyQt应用程序
    try:
        # 设置应用程序样式
        app = QApplication(sys.argv)
        app.setApplicationName("摔倒检测系统")

        # 设置全局字体
        font = QFont()
        font.setFamily("SimHei")  # 使用黑体字体以支持中文
        font.setPointSize(10)
        app.setFont(font)

        # 确保中文显示正常
        QApplication.setApplicationName("基于YOLOv11姿态估计的摔倒检测系统")

        # 创建并显示主窗口
        main_window = MainWindow(args)

        # 如果设置了全屏模式
        if args.fullscreen:
            main_window.showFullScreen()
        else:
            main_window.show()

        logger.info("主窗口创建成功，开始运行应用程序")
        logger.info("系统将自动收集准确率数据并显示在统计信息面板中")

        # 运行应用程序的主循环
        return app.exec_()

    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}", exc_info=True)
        print(f"错误: 应用程序启动失败 - {str(e)}")
        # 提供更详细的错误信息和解决方案建议
        if "CUDA out of memory" in str(e):
            print("\n错误提示: CUDA内存不足。建议:")
            print("1. 使用--device cpu参数切换到CPU模式")
            print("2. 如果您有GPU但内存不足，可以降低模型大小或输入尺寸")
        elif "No module named" in str(e):
            print("\n错误提示: 缺少必要的Python模块。请确保已安装所有依赖:")
            print("pip install opencv-python pyqt5 ultralytics pandas numpy")
        elif "模型加载失败" in str(e) or "Model load failed" in str(e):
            print("\n错误提示: 模型加载失败。建议:")
            print("1. 检查YOLOv11模型文件是否存在")
            print("2. 确保安装了最新版本的ultralytics库")
            print("3. 或使用系统自带的yolov8n-pose.pt模型")
        return 1
    finally:
        logger.info("=== 基于YOLOv11姿态估计的摔倒检测系统关闭 ===")


if __name__ == '__main__':
    # 运行主函数
    sys.exit(main())