"""
PyQt5主窗口模块

此模块实现了摔倒检测系统的图形用户界面
"""

import os
import sys
import logging
import cv2
import numpy as np
import winsound
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QTabWidget, QMessageBox, QSplitter, QTextEdit, QAction, QMenuBar, QToolBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QIcon

logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import YoloPoseDetector
from core.pose_analyzer import FallAnalyzer


class MainWindow(QMainWindow):
    """主窗口类"""

    # 信号定义
    update_status = pyqtSignal(str)

    def __init__(self, args):
        """初始化主窗口

        Args:
            args: 命令行参数
        """
        super().__init__()
        self.args = args

        # 初始化变量
        self.detector = None
        self.analyzer = None
        self.video_capture = None
        self.timer = None
        self.running = False
        self.current_frame = None
        self.current_image = None  # 用于存储静态图片
        self.processed_frame = None
        self.output_dir = "output"

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化UI
        self.init_ui()

        # 初始化检测器和分析器
        self.init_modules()

        # 连接信号槽
        self.update_status.connect(self.update_status_bar)

    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题和大小
        self.setWindowTitle("基于YOLOv11姿态估计的摔倒检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.create_tool_bar()

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建顶部控制区
        control_layout = self.create_control_panel()
        main_layout.addLayout(control_layout)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 创建左侧显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)

        # 原始图像显示
        self.raw_image_label = QLabel("原始图像")
        self.raw_image_label.setAlignment(Qt.AlignCenter)
        self.raw_image_label.setStyleSheet("background-color: black; color: white;")
        display_layout.addWidget(self.raw_image_label)

        # 处理后图像显示
        self.processed_image_label = QLabel("处理后图像")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("background-color: black; color: white;")
        display_layout.addWidget(self.processed_image_label)

        splitter.addWidget(display_widget)

        # 创建右侧信息面板
        info_widget = self.create_info_panel()
        splitter.addWidget(info_widget)

        # 设置分割器比例
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter, 1)

        # 创建状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label, 1)

        # 创建统计信息状态栏
        self.stats_label = QLabel("摔倒次数: 0")
        self.statusBar().addPermanentWidget(self.stats_label)

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        # 打开图片动作
        open_image_action = QAction("打开图片", self)
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)

        # 打开视频动作
        open_video_action = QAction("打开视频", self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)

        # 打开摄像头动作
        open_camera_action = QAction("打开摄像头", self)
        open_camera_action.triggered.connect(self.open_camera)
        file_menu.addAction(open_camera_action)

        # 保存结果动作
        save_result_action = QAction("保存结果", self)
        save_result_action.triggered.connect(self.save_result)
        file_menu.addAction(save_result_action)

        # 退出动作
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 设置菜单
        settings_menu = menubar.addMenu("设置")

        # 模型设置动作
        model_settings_action = QAction("模型设置", self)
        model_settings_action.triggered.connect(self.show_model_settings)
        settings_menu.addAction(model_settings_action)

        # 分析设置动作
        analysis_settings_action = QAction("分析设置", self)
        analysis_settings_action.triggered.connect(self.show_analysis_settings)
        settings_menu.addAction(analysis_settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        # 关于动作
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("工具栏")
        self.addToolBar(toolbar)

        # 打开图片按钮
        open_image_btn = QPushButton("打开图片")
        open_image_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_image_btn)

        # 打开视频按钮
        open_video_btn = QPushButton("打开视频")
        open_video_btn.clicked.connect(self.open_video)
        toolbar.addWidget(open_video_btn)

        # 打开摄像头按钮
        open_camera_btn = QPushButton("打开摄像头")
        open_camera_btn.clicked.connect(self.open_camera)
        toolbar.addWidget(open_camera_btn)

        toolbar.addSeparator()

        # 开始/暂停按钮
        self.start_stop_btn = QPushButton("开始")
        self.start_stop_btn.clicked.connect(self.toggle_processing)
        toolbar.addWidget(self.start_stop_btn)

        # 停止按钮
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        toolbar.addWidget(self.stop_btn)

    def create_control_panel(self):
        """创建控制面板"""
        layout = QHBoxLayout()

        # 置信度阈值滑块
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(int(self.args.conf_thres * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        conf_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel(f"{self.args.conf_thres:.2f}")
        conf_layout.addWidget(self.conf_value_label)

        # IOU阈值滑块
        iou_layout = QVBoxLayout()
        iou_layout.addWidget(QLabel("IOU阈值"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(int(self.args.iou_thres * 100))
        self.iou_slider.valueChanged.connect(self.on_iou_slider_changed)
        iou_layout.addWidget(self.iou_slider)

        self.iou_value_label = QLabel(f"{self.args.iou_thres:.2f}")
        iou_layout.addWidget(self.iou_value_label)

        # 添加到主布局
        layout.addLayout(conf_layout)
        layout.addLayout(iou_layout)
        layout.addStretch()

        return layout

    def create_info_panel(self):
        """创建信息面板"""
        # 创建标签页
        tab_widget = QTabWidget()

        # 检测信息标签页
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)

        # 检测结果文本框
        self.detection_info = QTextEdit()
        self.detection_info.setReadOnly(True)
        detection_layout.addWidget(QLabel("检测信息"))
        detection_layout.addWidget(self.detection_info, 1)

        tab_widget.addTab(detection_tab, "检测信息")

        # 统计信息标签页
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_info = QTextEdit()
        self.stats_info.setReadOnly(True)
        stats_layout.addWidget(QLabel("统计信息"))
        stats_layout.addWidget(self.stats_info, 1)

        tab_widget.addTab(stats_tab, "统计信息")

        # 参数设置标签页
        params_tab = QWidget()
        params_layout = QGridLayout(params_tab)

        # 髋-膝角度阈值
        params_layout.addWidget(QLabel("髋-膝角度阈值:"), 0, 0)
        self.hip_knee_threshold = QSpinBox()
        self.hip_knee_threshold.setRange(10, 170)
        self.hip_knee_threshold.setValue(45)
        self.hip_knee_threshold.valueChanged.connect(self.on_threshold_changed)
        params_layout.addWidget(self.hip_knee_threshold, 0, 1)

        # 肩-髋角度阈值
        params_layout.addWidget(QLabel("肩-髋角度阈值:"), 1, 0)
        self.shoulder_hip_threshold = QSpinBox()
        self.shoulder_hip_threshold.setRange(10, 80)
        self.shoulder_hip_threshold.setValue(30)
        self.shoulder_hip_threshold.valueChanged.connect(self.on_threshold_changed)
        params_layout.addWidget(self.shoulder_hip_threshold, 1, 1)

        # 高宽比阈值
        params_layout.addWidget(QLabel("高宽比阈值:"), 2, 0)
        self.hw_ratio_threshold = QDoubleSpinBox()
        self.hw_ratio_threshold.setRange(0.5, 3.0)
        self.hw_ratio_threshold.setSingleStep(0.1)
        self.hw_ratio_threshold.setValue(1.2)
        self.hw_ratio_threshold.valueChanged.connect(self.on_threshold_changed)
        params_layout.addWidget(self.hw_ratio_threshold, 2, 1)

        # 时间窗口大小
        params_layout.addWidget(QLabel("时间窗口大小:"), 3, 0)
        self.time_window_size = QSpinBox()
        self.time_window_size.setRange(1, 30)
        self.time_window_size.setValue(5)
        self.time_window_size.valueChanged.connect(self.on_time_window_changed)
        params_layout.addWidget(self.time_window_size, 3, 1)

        tab_widget.addTab(params_tab, "参数设置")

        return tab_widget

    def init_modules(self):
        """初始化检测器和分析器模块"""
        try:
            # 初始化分析器
            self.analyzer = FallAnalyzer(detector=None)  # 先初始化为None

            # 初始化检测器
            self.update_status.emit("正在加载YOLOv11姿态估计模型...")
            
            # 使用我们推荐的默认参数值，而不是直接使用命令行参数
            conf_thres = 0.4  # 降低置信度阈值以提高检测率
            iou_thres = 0.55  # 提高IOU阈值以减少重复检测
            
            self.detector = YoloPoseDetector(
                model_path=self.args.model,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                device=self.args.device
            )
            
            # 将检测器注册到分析器中，用于统计数据共享
            self.analyzer.register_detector(self.detector)
            
            # 更新UI中的滑块值，使其与实际使用的值一致
            self.conf_slider.setValue(int(conf_thres * 100))
            self.conf_value_label.setText(f"{conf_thres:.2f}")
            self.iou_slider.setValue(int(iou_thres * 100))
            self.iou_value_label.setText(f"{iou_thres:.2f}")

            self.update_status.emit(f"模型加载成功: {os.path.basename(self.args.model)}")
            logger.info("模块初始化完成")

        except Exception as e:
            self.update_status.emit(f"模块初始化失败: {str(e)}")
            logger.error(f"模块初始化失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"模块初始化失败: {str(e)}")
            # 即使初始化失败，也要确保在UI中能看到统计面板
            if hasattr(self, 'update_stats_info'):
                self.update_stats_info()

    def open_image(self):
        """打开图片文件进行分析"""
        # 停止当前处理
        self.stop_processing()
        
        # 重置当前图像变量
        self.current_image = None
        
        # 选择图片文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", ".", "图片文件 (*.jpg *.jpeg *.png *.bmp)")

        if file_path:
            try:
                # 读取图片
                image = cv2.imread(file_path)
                if image is None:
                    raise Exception("无法读取图片文件")

                self.update_status.emit(f"已打开图片文件: {os.path.basename(file_path)}")
                logger.info(f"已打开图片文件: {file_path}")
                
                # 保存当前图片
                self.current_image = image
                self.current_frame = image.copy()
                
                # 显示原图
                self.display_frame(self.current_frame, self.raw_image_label)
                
                # 提示用户点击开始按钮进行分析
                self.update_status.emit("请点击'开始'按钮进行摔倒分析")

            except Exception as e:
                self.update_status.emit(f"打开图片文件失败: {str(e)}")
                logger.error(f"打开图片文件失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"打开图片文件失败: {str(e)}")

    def open_video(self):
        """打开视频文件"""
        # 停止当前处理
        self.stop_processing()
        
        # 重置当前图像变量
        self.current_image = None

        # 选择视频文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", ".", "视频文件 (*.mp4 *.avi *.mov *.mkv)")

        if file_path:
            try:
                # 打开视频文件
                self.video_capture = cv2.VideoCapture(file_path)
                if not self.video_capture.isOpened():
                    raise Exception("无法打开视频文件")

                self.update_status.emit(f"已打开视频文件: {os.path.basename(file_path)}")
                logger.info(f"已打开视频文件: {file_path}")

                # 准备显示第一帧
                ret, frame = self.video_capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_frame(self.current_frame, self.raw_image_label)

            except Exception as e:
                self.update_status.emit(f"打开视频文件失败: {str(e)}")
                logger.error(f"打开视频文件失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"打开视频文件失败: {str(e)}")

    def open_camera(self):
        """打开摄像头"""
        # 停止当前处理
        self.stop_processing()
        
        # 重置当前图像变量
        self.current_image = None

        try:
            # 打开默认摄像头
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise Exception("无法打开摄像头")

            self.update_status.emit("摄像头已打开")
            logger.info("摄像头已打开")

            # 开始处理
            self.start_processing()

        except Exception as e:
            self.update_status.emit(f"打开摄像头失败: {str(e)}")
            logger.error(f"打开摄像头失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"打开摄像头失败: {str(e)}")

    def start_processing(self):
        """开始处理视频/摄像头"""
        if not self.video_capture:
            QMessageBox.warning(self, "警告", "请先打开视频文件或摄像头")
            return

        try:
            # 创建定时器
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(33)  # 约30fps

            self.running = True
            self.start_stop_btn.setText("暂停")
            self.update_status.emit("正在处理...")

        except Exception as e:
            self.update_status.emit(f"启动处理失败: {str(e)}")
            logger.error(f"启动处理失败: {str(e)}", exc_info=True)

    def stop_processing(self):
        """停止处理"""
        if self.timer:
            self.timer.stop()
            self.timer = None

        self.running = False
        self.start_stop_btn.setText("开始")

        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # 清空处理后的图像显示
        if hasattr(self, 'processed_image_label'):
            self.processed_image_label.setText("处理后图像")

        self.update_status.emit("处理已停止")
        logger.info("处理已停止")
        
    def process_single_image(self):
        """处理单张图片"""
        # 检查必要的组件是否存在
        if self.current_image is None or self.detector is None or self.analyzer is None:
            # 给出明确的错误提示
            if self.detector is None or self.analyzer is None:
                self.update_status.emit("错误: 检测器或分析器未初始化")
                QMessageBox.critical(self, "错误", "系统组件未正确初始化，请重启程序")
            return

        try:
            self.update_status.emit("正在分析图片...")
            
            # 检测人体姿态
            detections = self.detector.detect(self.current_image)

            # 分析姿态，判断是否摔倒
            analysis_results = self.analyzer.analyze_multi_poses(detections)

            # 绘制结果
            self.processed_frame = self.detector.draw_poses(self.current_image.copy(), detections)

            # 在处理后的图像上标记摔倒
            fall_detected = False
            for result in analysis_results:
                if result['analysis']['is_fall']:
                    fall_detected = True
                    bbox = result['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    # 绘制红色警告框
                    cv2.rectangle(self.processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # 添加警告文字（增强版：更大、更粗、带背景）
                    text = "摔倒!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 4
                    text_color = (0, 0, 255)  # 红色文字
                    bg_color = (255, 255, 255)  # 白色背景
                    
                    # 计算文字尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    
                    # 计算文字位置（确保不超出图像边界）
                    text_x = max(10, x1)
                    text_y = max(y1 - 10, text_height + 10)
                    
                    # 绘制文字背景
                    cv2.rectangle(self.processed_frame, 
                                 (text_x, text_y - text_height - baseline), 
                                 (text_x + text_width, text_y + baseline), 
                                 bg_color, -1)
                    
                    # 绘制文字
                    cv2.putText(self.processed_frame, text, (text_x, text_y),
                                font, font_scale, text_color, font_thickness)
            
            # 如果检测到摔倒，播放声音报警
            if fall_detected:
                winsound.Beep(1000, 500)  # 1000Hz频率，500ms持续时间
            
            # 显示处理后的图像
            self.display_frame(self.processed_frame, self.processed_image_label)

            # 更新信息面板
            self.update_detection_info(analysis_results)
            self.update_stats_info()
            
            # 显示分析结果消息
            if fall_detected:
                self.update_status.emit("分析完成: 检测到摔倒")
                QMessageBox.warning(self, "摔倒检测", "图片中检测到摔倒情况！")
            else:
                self.update_status.emit("分析完成: 未检测到摔倒")
                QMessageBox.information(self, "摔倒检测", "图片中未检测到摔倒情况。")

        except Exception as e:
            self.update_status.emit(f"处理图片失败: {str(e)}")
            logger.error(f"处理图片失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"处理图片失败: {str(e)}")

    def toggle_processing(self):
        """切换处理状态（开始/暂停）"""
        if self.running:
            if self.timer:
                self.timer.stop()
            self.running = False
            self.start_stop_btn.setText("开始")
            self.update_status.emit("处理已暂停")
        else:
            # 对于图片，直接处理一次；对于视频/摄像头，开始连续处理
            if hasattr(self, 'current_image') and self.current_image is not None:
                self.process_single_image()
            elif not self.video_capture:
                self.open_camera()
            else:
                self.start_processing()

    def process_frame(self):
        """处理单帧图像"""
        if self.video_capture is None or self.detector is None or self.analyzer is None:
            return

        try:
            # 读取帧
            ret, frame = self.video_capture.read()
            if not ret:
                # 视频播放完成
                self.update_status.emit("视频播放完成")
                self.stop_processing()
                return

            self.current_frame = frame

            # 检测人体姿态
            detections = self.detector.detect(frame)

            # 分析姿态，判断是否摔倒
            analysis_results = self.analyzer.analyze_multi_poses(detections)

            # 绘制结果
            self.processed_frame = self.detector.draw_poses(frame, detections)

            # 在处理后的图像上标记摔倒
            fall_detected = False
            for result in analysis_results:
                if result['analysis']['is_fall']:
                    fall_detected = True
                    bbox = result['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    # 绘制红色警告框
                    cv2.rectangle(self.processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # 添加警告文字（增强版：更大、更粗、带背景）
                    text = "摔倒!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 4
                    text_color = (0, 0, 255)  # 红色文字
                    bg_color = (255, 255, 255)  # 白色背景
                    
                    # 计算文字尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    
                    # 计算文字位置（确保不超出图像边界）
                    text_x = max(10, x1)
                    text_y = max(y1 - 10, text_height + 10)
                    
                    # 绘制文字背景
                    cv2.rectangle(self.processed_frame, 
                                 (text_x, text_y - text_height - baseline), 
                                 (text_x + text_width, text_y + baseline), 
                                 bg_color, -1)
                    
                    # 绘制文字
                    cv2.putText(self.processed_frame, text, (text_x, text_y),
                                font, font_scale, text_color, font_thickness)
            
            # 如果检测到摔倒，播放声音报警
            if fall_detected:
                winsound.Beep(1000, 500)  # 1000Hz频率，500ms持续时间

            # 显示图像
            self.display_frame(self.current_frame, self.raw_image_label)
            self.display_frame(self.processed_frame, self.processed_image_label)

            # 更新信息面板
            self.update_detection_info(analysis_results)
            self.update_stats_info()

        except Exception as e:
            self.update_status.emit(f"处理帧失败: {str(e)}")
            logger.error(f"处理帧失败: {str(e)}", exc_info=True)

    def display_frame(self, frame, label):
        """在QLabel中显示图像

        Args:
            frame: OpenCV图像 (BGR格式)
            label: QLabel控件
        """
        # 转换BGR为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸
        h, w, ch = rgb_frame.shape

        # 计算目标尺寸（保持比例）
        target_width = label.width()
        target_height = label.height()

        # 计算调整后的尺寸
        aspect_ratio = w / h
        if target_width / aspect_ratio <= target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # 调整图像大小
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height))

        # 创建QImage
        q_image = QImage(resized_frame.data, new_width, new_height, new_width * ch, QImage.Format_RGB888)

        # 显示在标签中
        label.setPixmap(QPixmap.fromImage(q_image))

    def update_detection_info(self, analysis_results):
        """更新检测信息面板"""
        info_text = f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        info_text += f"检测到的人数: {len(analysis_results)}\n\n"

        # 检查是否有摔倒
        any_fall = any(result['analysis']['is_fall'] for result in analysis_results)
        
        # 如果检测到摔倒，添加红色警告文字
        if any_fall:
            info_text += "<span style='color: red; font-weight: bold;'>检测到摔倒！</span>\n\n"

        for i, result in enumerate(analysis_results):
            info_text += f"人员 {i + 1}:\n"
            info_text += f"  置信度: {result['confidence']:.2f}\n"
            # 根据是否摔倒显示不同颜色的文字
            if result['analysis']['is_fall']:
                info_text += f"  是否摔倒: <span style='color: red; font-weight: bold;'>是</span>\n"
            else:
                info_text += f"  是否摔倒: 否\n"
            info_text += f"  分析置信度: {result['analysis']['confidence']:.2f}\n"
            info_text += "  特征值:\n"

            for key, value in result['analysis']['features'].items():
                # 确保值是Python原生类型，不是numpy数组
                if hasattr(value, 'tolist'):
                    value = value.tolist()
                elif hasattr(value, 'item'):
                    value = value.item()
                
                # 处理列表类型（如body_center）和数值类型
                if isinstance(value, list):
                    formatted_list = [f"{v:.2f}" for v in value]
                    info_text += f"    {key}: [{', '.join(formatted_list)}]\n"
                elif isinstance(value, (int, float)):
                    info_text += f"    {key}: {value:.2f}\n"
                else:
                    info_text += f"    {key}: {value}\n"
            info_text += "\n"

        self.detection_info.setHtml(info_text)

    def update_stats_info(self):
        """更新统计信息面板，包括摔倒统计和准确率数据"""
        info_text = f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 显示摔倒统计信息
        if self.analyzer:
            stats = self.analyzer.get_fall_statistics()
            info_text += f"===== 摔倒统计 =====\n"
            info_text += f"总摔倒次数: {stats['total_falls']}\n"
            info_text += f"最近1小时内摔倒次数: {stats['recent_falls']}\n"

            if stats['last_fall_time']:
                info_text += f"最后一次摔倒时间: {stats['last_fall_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            info_text += "\n"
        
        # 显示准确率统计信息
        if self.detector and hasattr(self.detector, 'accuracy_data'):
            accuracy_data = self.detector.accuracy_data
            info_text += f"===== 准确率统计 =====\n"
            
            # 计算总数和准确率指标
            total_samples = accuracy_data.get('true_positives', 0) + accuracy_data.get('false_positives', 0) + accuracy_data.get('false_negatives', 0)
            
            if total_samples > 0:
                info_text += f"总样本数: {total_samples}\n"
                info_text += f"真阳性 (正确检测到的摔倒): {accuracy_data.get('true_positives', 0)}\n"
                info_text += f"假阳性 (误报): {accuracy_data.get('false_positives', 0)}\n"
                info_text += f"假阴性 (漏报): {accuracy_data.get('false_negatives', 0)}\n"
                
                # 计算精确率、召回率和F1分数
                precision = accuracy_data.get('precision', 0)
                recall = accuracy_data.get('recall', 0)
                f1_score = accuracy_data.get('f1_score', 0)
                
                info_text += f"\n"
                info_text += f"精确率 (Precision): {precision:.2%}\n"
                info_text += f"召回率 (Recall): {recall:.2%}\n"
                info_text += f"F1分数: {f1_score:.2%}\n"
                
                # 显示检测器统计信息
                if hasattr(self.detector, 'total_frames'):
                    info_text += f"\n"
                    info_text += f"===== 检测统计 =====\n"
                    info_text += f"总帧数: {self.detector.total_frames}\n"
                    info_text += f"检测到的人数: {self.detector.detected_persons}\n"
            else:
                info_text += "尚未收集到足够的准确率数据...\n"
                info_text += "系统会在检测过程中自动计算准确率指标\n"
        
        self.stats_info.setText(info_text)
        
        # 更新状态栏中的摔倒次数
        if self.analyzer:
            stats = self.analyzer.get_fall_statistics()
            self.stats_label.setText(f"摔倒次数: {stats['total_falls']}")

    @pyqtSlot(str)
    def update_status_bar(self, message):
        """更新状态栏消息"""
        self.status_label.setText(message)

    def on_conf_slider_changed(self, value):
        """置信度滑块值变化处理"""
        conf_thres = value / 100.0
        self.conf_value_label.setText(f"{conf_thres:.2f}")

        if self.detector:
            self.detector.update_params(conf_thres=conf_thres)

    def on_iou_slider_changed(self, value):
        """IOU滑块值变化处理"""
        iou_thres = value / 100.0
        self.iou_value_label.setText(f"{iou_thres:.2f}")

        if self.detector:
            self.detector.update_params(iou_thres=iou_thres)

    def on_threshold_changed(self):
        """角度阈值变化处理"""
        if not self.analyzer:
            return

        thresholds = {
            'hip_knee_angle': self.hip_knee_threshold.value(),
            'shoulder_hip_angle': self.shoulder_hip_threshold.value(),
            'height_width_ratio': self.hw_ratio_threshold.value()
        }

        self.analyzer.update_thresholds(thresholds)

    def on_time_window_changed(self, value):
        """时间窗口大小变化处理"""
        if self.analyzer:
            self.analyzer.time_window = value

    def save_result(self):
        """保存当前处理结果"""
        if not self.processed_frame:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果")
            return

        # 生成保存文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"fall_detection_{timestamp}.jpg"

        # 选择保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", os.path.join(self.output_dir, default_filename), "图像文件 (*.jpg *.png)")

        if save_path:
            try:
                # 保存图像
                cv2.imwrite(save_path, self.processed_frame)
                self.update_status.emit(f"结果已保存: {os.path.basename(save_path)}")
                logger.info(f"结果已保存: {save_path}")

            except Exception as e:
                self.update_status.emit(f"保存结果失败: {str(e)}")
                logger.error(f"保存结果失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")

    def show_model_settings(self):
        """显示模型设置对话框"""
        # 这里可以实现更复杂的模型设置对话框
        QMessageBox.information(self, "模型设置", "当前模型: " + os.path.basename(self.args.model))

    def show_analysis_settings(self):
        """显示分析设置对话框"""
        # 这里可以实现更复杂的分析设置对话框
        QMessageBox.information(self, "分析设置", "分析参数可在右侧参数设置标签页中调整")

    def show_about(self):
        """显示关于对话框"""
        about_text = "基于YOLOv11姿态估计的摔倒检测系统\n"
        about_text += "版本: 1.0.0\n"
        about_text += "\n"
        about_text += "本系统使用YOLOv11姿态估计模型，通过分析人体关键点角度来检测摔倒事件。\n"
        about_text += "支持视频文件和摄像头实时检测。"

        QMessageBox.about(self, "关于", about_text)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_processing()
        event.accept()