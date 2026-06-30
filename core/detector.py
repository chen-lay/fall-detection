"""
YOLOv11姿态检测模块

此模块封装了YOLOv11姿态估计模型的加载和推理功能
"""

import os
import logging
import numpy as np
import torch
import cv2
from datetime import datetime
import pandas as pd


logger = logging.getLogger(__name__)


class YoloPoseDetector:
    """YOLOv11姿态检测器类，支持YOLOv11模型"""

    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.55, device='cuda:0'):
        """初始化YOLOv11姿态检测器

        Args:
            model_path: 模型权重文件路径
            conf_thres: 置信度阈值（降低以提高检测率）
            iou_thres: IOU阈值（提高以减少重复检测）
            device: 运行设备
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 设置设备
        self.device = device
        if not torch.cuda.is_available() or device == 'cpu':
            self.device = 'cpu'
            logger.warning("CUDA不可用，使用CPU模式")
        
        # 初始化模型
        self.model = None
        self.stride = None
        self.names = None
        self.load_model()

        # COCO数据集关键点顺序
        self.keypoints_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 关键点颜色配置
        self.keypoint_colors = {
            'nose': (255, 0, 0),        # 红色
            'eyes': (0, 255, 0),        # 绿色
            'ears': (0, 0, 255),        # 蓝色
            'shoulders': (255, 255, 0), # 青色
            'elbows': (255, 0, 255),    # 洋红色
            'wrists': (0, 255, 255),    # 黄色
            'hips': (128, 0, 128),      # 紫色
            'knees': (0, 128, 128),     # 蓝绿色
            'ankles': (128, 128, 0),    # 橄榄绿
        }
        
        # 性能统计
        self.detection_stats = {
            'total_frames': 0,
            'detected_persons': 0,
            'fall_detected': 0,
            'accuracy_data': []
        }
        
        # 确保runs文件夹存在
        self.runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
        if not os.path.exists(self.runs_dir):
            os.makedirs(self.runs_dir)
            logger.info(f"创建runs文件夹: {self.runs_dir}")

    def load_model(self):
        """加载YOLOv11姿态估计模型，优先尝试YOLOv11"""
        try:
            # 在方法内部导入ultralytics，避免模块加载时出错
            from ultralytics import YOLO

            # 检查模型文件是否存在
            if os.path.exists(self.model_path):
                logger.info(f"正在加载姿态估计模型: {self.model_path}")
                self.model = YOLO(self.model_path, task='pose')
            else:
                # 尝试直接加载YOLOv11模型
                if 'yolov11' in self.model_path.lower():
                    model_name = os.path.basename(self.model_path)
                    logger.warning(f"模型文件不存在: {self.model_path}，尝试下载并加载YOLOv11模型...")
                    try:
                        # 尝试自动下载YOLOv11模型
                        # 首先检查是否安装了支持YOLOv11的ultralytics版本
                        import pkg_resources
                        ultralytics_version = pkg_resources.get_distribution('ultralytics').version
                        logger.info(f"当前ultralytics版本: {ultralytics_version}")
                        
                        # 尝试不同的YOLOv11模型名称格式
                        yolo11_models = ['yolov11n-pose', 'yolo11n-pose.pt', 'n', 'yolo11-pose']
                        loaded = False
                        
                        for model_name in yolo11_models:
                            try:
                                logger.info(f"尝试加载模型: {model_name}")
                                self.model = YOLO(model_name, task='pose')
                                loaded = True
                                logger.info(f"成功加载模型: {model_name}")
                                break
                            except Exception as e:
                                logger.warning(f"加载模型 {model_name} 失败: {str(e)}")
                                continue
                        
                        if not loaded:
                            raise Exception("无法加载任何YOLOv11模型")
                    except Exception as e:
                        logger.warning(f"YOLOv11模型加载失败: {str(e)}")
                        logger.warning("回退到YOLOv8模型")
                        # 回退到YOLOv8模型
                        if os.path.exists('yolov8n-pose.pt'):
                            logger.info("加载YOLOv8姿态模型: yolov8n-pose.pt")
                            self.model = YOLO('yolov8n-pose.pt', task='pose')
                        else:
                            # 尝试从ultralytics自动下载YOLOv8模型
                            logger.info("尝试自动下载YOLOv8模型...")
                            try:
                                self.model = YOLO('yolov8n-pose.pt', task='pose')
                                logger.info("成功下载并加载YOLOv8模型")
                            except Exception as inner_e:
                                logger.error(f"无法下载YOLOv8模型: {str(inner_e)}")
                                raise Exception("无法加载任何姿态估计模型")
                else:
                    logger.warning(f"模型文件不存在: {self.model_path}，尝试使用官方预训练模型...")
                    # 使用官方预训练模型
                    self.model = YOLO('yolov8n-pose.pt', task='pose')
                    logger.info(f"正在加载官方预训练模型: yolov8n-pose.pt")
            
            # 将模型移动到指定设备
            self.model.to(self.device)

            # 获取模型信息
            # 新版本的ultralytics可能没有直接暴露stride属性，我们使用默认值
            self.stride = 32  # YOLO模型通常使用32的步长
            # 尝试获取类别名称，如果不存在则使用默认名称
            try:
                self.names = self.model.names
            except AttributeError:
                self.names = {0: 'person'}  # 默认为人

            logger.info(f"模型加载成功，运行设备: {self.device}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            raise

    def detect(self, image):
        """检测图像中的人体姿态，添加性能统计

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            detections: 检测结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")

        detections = []
        
        try:
            # 增加帧计数
            self.detection_stats['total_frames'] += 1
            
            # 运行推理 - 使用增强的参数设置
            results = self.model(image,
                                 conf=self.conf_thres,
                                 iou=self.iou_thres,
                                 device=self.device,
                                 verbose=False,
                                 imgsz=640,  # 统一输入尺寸以提高性能
                                 augment=True)  # 启用数据增强以提高检测率

            # 解析结果
            for result in results:
                # 获取关键点信息
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    # 确保结果不为空
                    if len(result.boxes) == 0 or len(result.keypoints) == 0:
                        logger.debug("未检测到人体")
                        continue
                        
                    try:
                        # 转换为numpy数组
                        kpts = result.keypoints.data.cpu().numpy()
                        boxes = result.boxes.xyxy.data.cpu().numpy()
                        confs = result.boxes.conf.data.cpu().numpy()
                        
                        # 检查数组大小是否匹配
                        if len(kpts) != len(boxes) or len(kpts) != len(confs):
                            logger.warning("检测结果数组大小不匹配")
                            continue
                            
                        # 处理每个人体
                        for i in range(len(kpts)):
                            # 对低置信度关键点进行平滑处理
                            processed_kpts = []
                            for kp in kpts[i]:
                                if len(kp) >= 3 and kp[2] < 0.3:
                                    kp[2] = 0.0  # 过滤掉极低置信度的点
                                processed_kpts.append(kp)
                                
                            detection = {
                                'bbox': boxes[i],  # [x1, y1, x2, y2]
                                'confidence': confs[i],
                                'keypoints': np.array(processed_kpts),  # 17个关键点，每个关键点包含[x, y, 置信度]
                                'class_id': 0,  # 人体类别
                                'id': i,
                                'timestamp': datetime.now()
                            }
                            detections.append(detection)
                    except Exception as inner_e:
                        logger.error(f"解析检测结果时出错: {str(inner_e)}")
                        continue
            
            # 更新检测统计
            self.detection_stats['detected_persons'] += len(detections)

        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}", exc_info=True)
            return []
        
        return detections

    def draw_poses(self, image, detections, draw_bbox=True, draw_keypoints=True, draw_skeleton=True):
        """在图像上绘制姿态检测结果

        Args:
            image: 输入图像
            detections: 检测结果
            draw_bbox: 是否绘制边界框
            draw_keypoints: 是否绘制关键点
            draw_skeleton: 是否绘制骨架

        Returns:
            output_image: 绘制结果后的图像
        """
        output_image = image.copy()

        # 关键点颜色配置
        keypoint_colors = {
            'nose': (255, 0, 0),
            'eyes': (0, 0, 255),
            'ears': (0, 255, 0),
            'shoulders': (255, 255, 0),
            'elbows': (255, 0, 255),
            'wrists': (0, 255, 255),
            'hips': (255, 128, 0),
            'knees': (128, 255, 0),
            'ankles': (0, 128, 255)
        }

        # 骨架连接配置
        skeleton_connections = [
            (5, 6),  # 左右肩
            (5, 7),  # 左肩到左肘
            (7, 9),  # 左肘到左手腕
            (6, 8),  # 右肩到右肘
            (8, 10),  # 右肘到右手腕
            (5, 11),  # 左肩到左髋
            (6, 12),  # 右肩到右髋
            (11, 12),  # 左右髋
            (11, 13),  # 左髋到左膝
            (13, 15),  # 左膝到左脚踝
            (12, 14),  # 右髋到右膝
            (14, 16)  # 右膝到右脚踝
        ]

        # 为每个人体绘制结果
        for detection in detections:
            bbox = detection['bbox']
            keypoints = detection['keypoints']
            confidence = detection['confidence']

            # 绘制边界框
            if draw_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制置信度
                label = f'Person: {confidence:.2f}'
                cv2.putText(output_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制骨架连接
            if draw_skeleton:
                for start, end in skeleton_connections:
                    if start < len(keypoints) and end < len(keypoints):
                        s_x, s_y, s_conf = keypoints[start]
                        e_x, e_y, e_conf = keypoints[end]

                        # 只有当两个关键点都有足够的置信度时才绘制连接线
                        if s_conf > 0.5 and e_conf > 0.5:
                            cv2.line(output_image, (int(s_x), int(s_y)), (int(e_x), int(e_y)),
                                     (255, 0, 255), 2)

            # 绘制关键点
            if draw_keypoints:
                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.5:  # 只绘制置信度高的关键点
                        # 根据关键点类型选择颜色
                        if i == 0:  # nose
                            color = keypoint_colors['nose']
                        elif 1 <= i <= 4:  # eyes and ears
                            color = keypoint_colors['eyes'] if i <= 3 else keypoint_colors['ears']
                        elif 5 <= i <= 6:  # shoulders
                            color = keypoint_colors['shoulders']
                        elif 7 <= i <= 8:  # elbows
                            color = keypoint_colors['elbows']
                        elif 9 <= i <= 10:  # wrists
                            color = keypoint_colors['wrists']
                        elif 11 <= i <= 12:  # hips
                            color = keypoint_colors['hips']
                        elif 13 <= i <= 14:  # knees
                            color = keypoint_colors['knees']
                        elif 15 <= i <= 16:  # ankles
                            color = keypoint_colors['ankles']
                        else:
                            color = (255, 255, 255)

                        # 绘制关键点
                        cv2.circle(output_image, (int(x), int(y)), 5, color, -1)

                        # 可选：绘制关键点索引
                        # cv2.putText(output_image, str(i), (int(x)+10, int(y)),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return output_image

    def update_params(self, conf_thres=None, iou_thres=None):
        """更新检测参数

        Args:
            conf_thres: 新的置信度阈值
            iou_thres: 新的IOU阈值
        """
        if conf_thres is not None:
            self.conf_thres = conf_thres
            logger.info(f"更新置信度阈值: {conf_thres}")

        if iou_thres is not None:
            self.iou_thres = iou_thres
            logger.info(f"更新IOU阈值: {iou_thres}")
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)
            
    def update_detection_stats(self, is_fall_detected):
        """更新检测统计信息
        
        Args:
            is_fall_detected: 是否检测到摔倒
        """
        if is_fall_detected:
            self.detection_stats['fall_detected'] += 1
            
    def log_accuracy_data(self, true_positive, false_positive, false_negative):
        """记录准确率数据
        
        Args:
            true_positive: 真阳性数量
            false_positive: 假阳性数量
            false_negative: 假阴性数量
        """
        accuracy_record = {
            'timestamp': datetime.now(),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
        self.detection_stats['accuracy_data'].append(accuracy_record)
        
        # 当积累到一定数量时保存到文件
        if len(self.detection_stats['accuracy_data']) % 10 == 0:
            self.save_accuracy_stats()
            
    def save_accuracy_stats(self):
        """保存准确率统计到文件"""
        try:
            stats_file = os.path.join(self.runs_dir, 'detection_accuracy.csv')
            df = pd.DataFrame(self.detection_stats['accuracy_data'])
            
            # 计算准确率指标
            if not df.empty:
                total_tp = df['true_positive'].sum()
                total_fp = df['false_positive'].sum()
                total_fn = df['false_negative'].sum()
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存详细记录
                df.to_csv(stats_file, index=False)
                
                # 保存汇总统计
                summary_file = os.path.join(self.runs_dir, 'detection_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 摔倒检测系统准确率统计 ===\n")
                    f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总帧数: {self.detection_stats['total_frames']}\n")
                    f.write(f"检测到的人数: {self.detection_stats['detected_persons']}\n")
                    f.write(f"检测到的摔倒次数: {self.detection_stats['fall_detected']}\n")
                    f.write(f"精确率 (Precision): {precision:.4f}\n")
                    f.write(f"召回率 (Recall): {recall:.4f}\n")
                    f.write(f"F1分数: {f1_score:.4f}\n")
                
                logger.info(f"准确率统计已保存到 {stats_file} 和 {summary_file}")
                logger.info(f"当前准确率: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
                
        except Exception as e:
            logger.error(f"保存准确率统计失败: {str(e)}", exc_info=True)