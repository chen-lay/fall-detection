"""
姿态分析模块

此模块用于分析人体关键点，判断是否发生摔倒
"""

import numpy as np
import logging
import statistics
from datetime import datetime
import winsound  # 添加声音报警模块

logger = logging.getLogger(__name__)


class FallAnalyzer:
    """摔倒分析器类 - 增强版"""

    def __init__(self, detector=None):
        """初始化摔倒分析器
        
        Args:
            detector: YoloPoseDetector实例，用于更新统计信息
        """
        # 优化后的关键角度阈值配置 - 调整阈值以提高检测灵敏度
        self.angle_thresholds = {
            'hip_knee_angle': 50,          # 增加髋-膝角度阈值以提高检测灵敏度
            'knee_ankle_angle': 60,        # 增加膝-踝角度阈值以提高检测灵敏度
            'shoulder_hip_angle': 40,      # 增加肩-髋角度阈值以提高检测灵敏度
            'height_width_ratio': 1.3,     # 增加高度/宽度比例阈值以提高检测率
            'center_y_position': 0.65,     # 调整人体中心相对位置阈值
            'velocity_change': 0.10        # 调整速度变化阈值
        }

        # 优化后的特征权重配置 - 调整权重以提高检测准确性
        self.feature_weights = {
            'height_width_ratio': 0.35,    # 稍微降低高度/宽度比例权重
            'angle_score': 0.40,           # 增加角度分析权重
            'position_score': 0.15,        # 降低位置分析权重
            'movement_score': 0.10         # 保持运动分析权重
        }
        
        # 时间窗口配置（用于平滑检测结果）
        self.time_window = 12  # 增加检测窗口大小以更好地分析运动
        self.fall_history = []  # 摔倒检测历史
        self.person_history = {}  # 跟踪每个人的历史位置

        # 摔倒事件记录
        self.fall_events = []
        
        # 检测器引用（用于更新统计）
        self.detector = detector
        
        # 评估统计信息 - 增强的数据收集
        self.evaluation_stats = {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
        # 初始化评估数据收集
        self.current_frame_ground_truth = None  # 用于评估的真实标签
        
        # 报警相关设置
        self.last_alarm_time = 0
        self.alarm_cooldown = 2  # 报警冷却时间（秒），避免持续报警
        
        logger.info("摔倒分析器初始化完成，已加载优化的检测参数")
        logger.info(f"初始检测阈值配置: {self.angle_thresholds}")
        logger.info(f"特征权重配置: {self.feature_weights}")

    def calculate_angle(self, a, b, c):
        """计算三点之间的夹角

        Args:
            a: 第一个点 [x, y]
            b: 中间点 [x, y]
            c: 第三个点 [x, y]

        Returns:
            angle: 角度值（0-180度）
        """
        # 计算向量
        ba = a - b
        bc = c - b

        # 计算夹角余弦值
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

        # 限制余弦值范围在[-1, 1]之间
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # 转换为角度
        angle = np.arccos(cosine_angle) * 180.0 / np.pi

        return angle

    def trigger_alarm(self):
        """触发声音报警
        
        只有在冷却时间过后才会触发报警，避免持续报警造成干扰
        """
        import time
        current_time = time.time()
        
        # 检查是否在冷却期内
        if current_time - self.last_alarm_time > self.alarm_cooldown:
            try:
                # 使用winsound播放警报声
                winsound.Beep(1000, 500)  # 1000Hz, 500ms
                self.last_alarm_time = current_time
                logger.info("触发摔倒报警")
            except Exception as e:
                logger.error(f"报警失败: {str(e)}")

    def calculate_distance(self, a, b):
        """计算两点之间的欧氏距离

        Args:
            a: 第一个点 [x, y]
            b: 第二个点 [x, y]

        Returns:
            distance: 欧氏距离
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def trigger_alarm(self):
        """触发声音报警"""
        current_time = datetime.now()
        # 检查是否在冷却期内
        if (current_time - self.last_alarm_time).total_seconds() > self.alarm_cooldown:
            try:
                # 播放系统警报声
                winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
                self.last_alarm_time = current_time
                logger.info("声音报警已触发")
            except Exception as e:
                logger.error(f"报警失败: {str(e)}")

    def _calculate_body_center(self, keypoints):
        """计算人体中心
        
        Args:
            keypoints: 人体关键点坐标
            
        Returns:
            body_center: 人体中心坐标
        """
        # 使用肩部和髋部关键点计算人体中心
        shoulders = [keypoints[5][:2], keypoints[6][:2]]  # 左右肩
        hips = [keypoints[11][:2], keypoints[12][:2]]      # 左右髋
        
        # 过滤掉置信度过低的关键点
        valid_points = []
        for point in shoulders + hips:
            if point is not None:
                valid_points.append(point)
        
        if not valid_points:
            return None
            
        return np.mean(valid_points, axis=0)
    
    def _calculate_body_dimensions(self, keypoints):
        """计算人体高度和宽度
        
        Args:
            keypoints: 人体关键点坐标
            
        Returns:
            height: 人体高度
            width: 人体宽度
        """
        # 确保keypoints是形状为(17, 3)的数组
        logger.debug(f"原始keypoints形状: {keypoints.shape}, 维度: {keypoints.ndim}")
        
        if keypoints.ndim == 3:
            if keypoints.shape[0] == 1:
                keypoints = keypoints[0]  # 处理形状为(1, 17, 3)的情况
            elif keypoints.shape[2] == 1:
                keypoints = keypoints.squeeze(2)  # 处理形状为(17, 3, 1)的情况
        
        # 处理所有可能的维度问题
        if keypoints.ndim != 2:
            logger.warning(f"keypoints维度异常: {keypoints.ndim}, 形状: {keypoints.shape}")
            return 0, 0
        
        # 提取关键部位的关键点
        def get_point(idx):
            if idx < keypoints.shape[0]:
                x, y, conf = keypoints[idx]  # 明确提取三个值
                if isinstance(conf, np.ndarray):
                    conf = conf.item()  # 转换为标量
                return keypoints[idx][:2] if conf > 0.2 else None
            return None
        
        nose = get_point(0)
        left_shoulder = get_point(5)
        right_shoulder = get_point(6)
        left_hip = get_point(11)
        right_hip = get_point(12)
        left_ankle = get_point(15)
        right_ankle = get_point(16)
        
        # 计算人体高度（头顶到脚踝的距离）
        head_points = [nose] if nose is not None else []
        ankle_points = [left_ankle, right_ankle] if left_ankle is not None or right_ankle is not None else []
        
        height = 0
        if head_points and ankle_points:
            max_head = np.max([p[1] for p in head_points if p is not None])
            min_ankle = np.min([p[1] for p in ankle_points if p is not None])
            height = min_ankle - max_head
        
        # 计算人体宽度（肩部或髋部的宽度）
        width = 0
        if left_shoulder is not None and right_shoulder is not None:
            width = np.abs(left_shoulder[0] - right_shoulder[0]) * 1.5  # 扩大1.5倍以包含整个身体
        elif left_hip is not None and right_hip is not None:
            width = np.abs(left_hip[0] - right_hip[0]) * 1.5
        
        return max(height, 1), max(width, 1)  # 确保至少为1
    
    def _calculate_hip_knee_angle(self, keypoints):
        """计算髋-膝角度
        
        Args:
            keypoints: 人体关键点坐标
            
        Returns:
            avg_angle: 平均髋-膝角度
        """
        angles = []
        
        # 计算左侧髋-膝-踝角度
        left_hip = keypoints[11][:2] if keypoints[11][2] > 0.2 else None
        left_knee = keypoints[13][:2] if keypoints[13][2] > 0.2 else None
        left_ankle = keypoints[15][:2] if keypoints[15][2] > 0.2 else None
        
        if left_hip is not None and left_knee is not None:
            # 如果没有脚踝点，使用膝盖下方的点
            if left_ankle is None:
                left_ankle = left_knee + np.array([0, 50])
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            angles.append(angle)
        
        # 计算右侧髋-膝-踝角度
        right_hip = keypoints[12][:2] if keypoints[12][2] > 0.2 else None
        right_knee = keypoints[14][:2] if keypoints[14][2] > 0.2 else None
        right_ankle = keypoints[16][:2] if keypoints[16][2] > 0.2 else None
        
        if right_hip is not None and right_knee is not None:
            # 如果没有脚踝点，使用膝盖下方的点
            if right_ankle is None:
                right_ankle = right_knee + np.array([0, 50])
            angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            angles.append(angle)
        
        return np.mean(angles) if angles else 180
    
    def _calculate_shoulder_hip_angle(self, keypoints):
        """计算肩-髋角度
        
        Args:
            keypoints: 人体关键点坐标
            
        Returns:
            angle: 肩-髋角度
        """
        # 计算肩部中心
        left_shoulder = keypoints[5][:2] if keypoints[5][2] > 0.2 else None
        right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.2 else None
        
        # 计算髋部中心
        left_hip = keypoints[11][:2] if keypoints[11][2] > 0.2 else None
        right_hip = keypoints[12][:2] if keypoints[12][2] > 0.2 else None
        
        shoulders = []
        if left_shoulder is not None:
            shoulders.append(left_shoulder)
        if right_shoulder is not None:
            shoulders.append(right_shoulder)
        
        hips = []
        if left_hip is not None:
            hips.append(left_hip)
        if right_hip is not None:
            hips.append(right_hip)
        
        if not shoulders or not hips:
            return 0
            
        shoulder_center = np.mean(shoulders, axis=0)
        hip_center = np.mean(hips, axis=0)
        
        # 计算与垂直方向的夹角
        vertical_line = shoulder_center + np.array([0, -100])  # 垂直向上的线
        return self.calculate_angle(vertical_line, shoulder_center, hip_center)
    
    def _analyze_position(self, body_center, image_shape):
        """分析人体位置
        
        Args:
            body_center: 人体中心坐标
            image_shape: 图像形状
            
        Returns:
            position_score: 位置评分
        """
        if body_center is None or image_shape is None:
            return 0.0
            
        # 计算人体中心相对于图像高度的比例
        relative_y = body_center[1] / image_shape[0]
        
        # 摔倒时人体中心通常较低
        if relative_y > self.angle_thresholds['center_y_position']:
            return (relative_y - self.angle_thresholds['center_y_position']) / (1.0 - self.angle_thresholds['center_y_position'])
        return 0.0
    
    def _analyze_motion(self, person_id, body_center):
        """分析人体运动
        
        Args:
            person_id: 人体ID
            body_center: 人体中心坐标
            
        Returns:
            motion_score: 运动评分
        """
        if person_id is None or body_center is None:
            return 0.0
            
        # 初始化历史记录
        if person_id not in self.person_history:
            self.person_history[person_id] = []
        
        # 记录当前位置
        self.person_history[person_id].append({
            'center': body_center,
            'timestamp': datetime.now()
        })
        
        # 保留最近的历史记录
        if len(self.person_history[person_id]) > self.time_window:
            self.person_history[person_id] = self.person_history[person_id][-self.time_window:]
        
        # 如果历史记录不足，返回0
        if len(self.person_history[person_id]) < 3:
            return 0.0
        
        # 计算垂直方向的速度变化
        velocities = []
        for i in range(1, len(self.person_history[person_id])):
            prev = self.person_history[person_id][i-1]
            curr = self.person_history[person_id][i]
            
            # 计算时间差（秒）
            time_diff = (curr['timestamp'] - prev['timestamp']).total_seconds()
            if time_diff < 0.01:
                continue
            
            # 计算垂直方向的速度
            dy = curr['center'][1] - prev['center'][1]
            velocity = dy / time_diff
            velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # 计算速度的标准差（衡量运动的突然变化）
        velocity_std = np.std(velocities)
        
        # 摔倒时通常有较大的速度变化
        return min(1.0, velocity_std / 500.0)  # 归一化到0-1范围
    
    def _calculate_fall_score(self, height_width_ratio, hip_knee_angle, shoulder_hip_angle, position_score, motion_score):
        """计算摔倒综合评分
        
        Args:
            height_width_ratio: 高度宽度比
            hip_knee_angle: 髋-膝角度
            shoulder_hip_angle: 肩-髋角度
            position_score: 位置评分
            motion_score: 运动评分
            
        Returns:
            fall_score: 摔倒综合评分
        """
        # 计算角度评分
        angle_score = 0.0
        
        # 髋-膝角度评分（角度越小，评分越高）
        if hip_knee_angle < self.angle_thresholds['hip_knee_angle']:
            angle_score += (self.angle_thresholds['hip_knee_angle'] - hip_knee_angle) / self.angle_thresholds['hip_knee_angle']
        
        # 肩-髋角度评分（角度越大，评分越高）
        if shoulder_hip_angle > 90 - self.angle_thresholds['shoulder_hip_angle']:
            angle_score += (shoulder_hip_angle - (90 - self.angle_thresholds['shoulder_hip_angle'])) / self.angle_thresholds['shoulder_hip_angle']
        
        angle_score = min(1.0, angle_score / 2.0)  # 归一化到0-1范围
        
        # 高度宽度比评分
        hw_ratio_score = 0.0
        if height_width_ratio < self.angle_thresholds['height_width_ratio']:
            hw_ratio_score = 1.0 - (height_width_ratio / self.angle_thresholds['height_width_ratio'])
        
        # 综合加权评分
        fall_score = (
            self.feature_weights['height_width_ratio'] * hw_ratio_score +
            self.feature_weights['angle_score'] * angle_score +
            self.feature_weights['position_score'] * position_score +
            self.feature_weights['movement_score'] * motion_score
        )
        
        return fall_score
    
    def _get_dynamic_threshold(self, features):
        """获取动态阈值，根据当前姿态特征调整阈值
        
        Args:
            features: 姿态特征字典
            
        Returns:
            threshold: 动态阈值
        """
        # 基础阈值，降低阈值以提高摔倒检测的灵敏度
        base_threshold = 0.3
        
        # 根据特征调整阈值
        if 'height_width_ratio' in features and features['height_width_ratio'] < 0.8:
            # 高度宽度比非常小，降低阈值以减少漏检
            base_threshold = 0.55
        elif 'shoulder_hip_angle' in features and features['shoulder_hip_angle'] > 80:
            # 肩-髋角度很大，降低阈值
            base_threshold = 0.60
        
        return base_threshold
    
    def _update_history(self, analysis_result, person_id):
        """更新历史记录
        
        Args:
            analysis_result: 分析结果
            person_id: 人体ID
        """
        # 更新摔倒检测历史
        self.fall_history.append({
            'timestamp': datetime.now(),
            'is_fall': analysis_result['is_fall'],
            'confidence': analysis_result['confidence'],
            'person_id': person_id
        })
        
        # 保留最近的历史记录
        if len(self.fall_history) > self.time_window * 2:
            self.fall_history = self.fall_history[-self.time_window * 2:]
    
    def analyze_pose(self, keypoints, person_id=None, image_shape=None):
        """分析单个人体姿态，判断是否摔倒

        Args:
            keypoints: 人体关键点坐标
            person_id: 人体ID，用于跟踪
            image_shape: 图像形状(h, w)，用于位置分析

        Returns:
            analysis_result: 分析结果
        """
        # 初始化分析结果
        analysis_result = {
            'is_fall': False,
            'confidence': 0.0,
            'features': {}
        }

        try:
            # 确保关键点数量正确
            if len(keypoints) < 17:
                logger.warning("关键点数量不足，无法进行完整的摔倒分析")
                return analysis_result

            # 计算人体中心
            body_center = self._calculate_body_center(keypoints)
            analysis_result['features']['body_center'] = body_center

            # 计算人体高度和宽度（用于判断是否摔倒）
            height, width = self._calculate_body_dimensions(keypoints)
            analysis_result['features']['body_height'] = height
            analysis_result['features']['body_width'] = width

            # 计算高度宽度比（用于判断是否摔倒）
            if width > 0:
                height_width_ratio = height / width
                analysis_result['features']['height_width_ratio'] = height_width_ratio
            else:
                height_width_ratio = 0

            # 计算髋-膝-踝角度（用于判断是否摔倒）
            hip_knee_angle = self._calculate_hip_knee_angle(keypoints)
            analysis_result['features']['hip_knee_angle'] = hip_knee_angle

            # 计算肩-髋-膝角度（用于判断是否摔倒）
            shoulder_hip_angle = self._calculate_shoulder_hip_angle(keypoints)
            analysis_result['features']['shoulder_hip_angle'] = shoulder_hip_angle

            # 位置分析（判断人体在图像中的垂直位置）
            position_score = self._analyze_position(body_center, image_shape)
            analysis_result['features']['position_score'] = position_score

            # 历史运动分析（判断是否有突然的垂直方向变化）
            motion_score = self._analyze_motion(person_id, body_center)
            analysis_result['features']['motion_score'] = motion_score

            # 综合评分（判断是否摔倒）
            fall_score = self._calculate_fall_score(
                height_width_ratio, hip_knee_angle, shoulder_hip_angle, position_score, motion_score
            )
            analysis_result['confidence'] = fall_score

            # 动态阈值判断
            threshold = self._get_dynamic_threshold(analysis_result['features'])
            analysis_result['is_fall'] = fall_score > threshold

            # 时序验证：连续多帧检测到摔倒才确认
            if analysis_result['is_fall'] and person_id:
                # 检查最近几帧的检测结果
                recent_results = [h for h in self.fall_history 
                                if h['person_id'] == person_id 
                                and (datetime.now() - h['timestamp']).total_seconds() < 2.0]
                
                # 计算连续摔倒帧数
                consecutive_falls = 0
                for result in reversed(recent_results):
                    if result['is_fall']:
                        consecutive_falls += 1
                    else:
                        break
                
                # 需要连续3帧以上检测到摔倒才确认
                analysis_result['is_fall'] = consecutive_falls >= 3
            
            # 如果检测到摔倒，触发报警
            if analysis_result['is_fall']:
                self.trigger_alarm()
                
                # 记录摔倒事件
                fall_event = {
                    'timestamp': datetime.now(),
                    'person_id': person_id,
                    'confidence': analysis_result['confidence'],
                    'features': analysis_result['features']
                }
                self.fall_events.append(fall_event)
        
            # 更新历史记录
            self._update_history(analysis_result, person_id)

        except Exception as e:
            logger.error(f"姿态分析出错: {str(e)}", exc_info=True)

        return analysis_result

    def analyze_multi_poses(self, detections, image_shape=None):
        """分析多个人体的姿态

        Args:
            detections: 包含多个人体检测结果的列表
            image_shape: 图像形状(h, w)，用于位置分析

        Returns:
            analysis_results: 每个人体的分析结果
        """
        analysis_results = []
        current_time = datetime.now()

        # 确保detections是列表类型
        if not isinstance(detections, list):
            detections = []
            logger.warning("检测结果格式错误，将按空列表处理")

        for i, detection in enumerate(detections):
            try:
                # 使用人体ID进行历史跟踪，并传入图像形状
                result = self.analyze_pose(
                    detection['keypoints'],
                    person_id=detection.get('id', i),
                    image_shape=image_shape
                )
                
                # 将分析结果与检测结果合并
                analysis_result = {
                    **detection,
                    'analysis': result
                }
                analysis_results.append(analysis_result)
                
                # 更新评估统计
                self.evaluation_stats['total_predictions'] += 1
                if result['is_fall'] and result['confidence'] > 0.7:  # 高置信度的检测作为正确预测
                    self.evaluation_stats['correct_predictions'] += 1
                
                # 尝试获取人体ID或创建临时ID用于跟踪
                person_id = detection.get('id', str(int(hash(str(detection.get('bbox', i)))) % 1000))
                
                # 更新人体历史记录
                if person_id not in self.person_history:
                    self.person_history[person_id] = []
                
                # 记录当前状态
                self.person_history[person_id].append({
                    'timestamp': current_time,
                    'center': result['features']['body_center'],
                    'bbox': detection.get('bbox'),
                    'is_fall': result['is_fall'],
                    'confidence': result['confidence']
                })
                
                # 保留最近的历史记录
                if len(self.person_history[person_id]) > self.time_window:
                    self.person_history[person_id] = self.person_history[person_id][-self.time_window:]
                
                # 优化漏检检测逻辑 - 更全面地分析潜在的摔倒特征
                if not result['is_fall']:
                    # 检查关键点质量
                    valid_keypoints = [k for k in detection['keypoints'] if k[2] > 0.3]
                    if len(valid_keypoints) >= 5:
                        features = result['features']
                        potential_fall = False
                        
                        # 多特征组合检测漏检
                        height_width_ratio = features.get('height_width_ratio', 2.0)
                        hip_knee_angle = features.get('hip_knee_angle', 180)
                        shoulder_hip_angle = features.get('shoulder_hip_angle', 0)
                        
                        # 高度/宽度比例接近摔倒阈值
                        if height_width_ratio < 1.5:
                            potential_fall = True
                            logger.debug(f"潜在漏检 - 高宽比: {height_width_ratio:.2f}")
                        
                        # 髋-膝角度较小
                        if hip_knee_angle < 60:
                            potential_fall = True
                            logger.debug(f"潜在漏检 - 髋-膝角度: {hip_knee_angle:.2f}°")
                        
                        # 肩-髋角度较大（接近水平）
                        if shoulder_hip_angle > 70:
                            potential_fall = True
                            logger.debug(f"潜在漏检 - 肩-髋角度: {shoulder_hip_angle:.2f}°")
                        
                        # 垂直位置较低
                        if 'position_score' in features and features['position_score'] > 0.5:
                            potential_fall = True
                            logger.debug(f"潜在漏检 - 位置评分: {features['position_score']:.2f}")
                        
                        # 综合评分接近阈值
                        if result['confidence'] > 0.55:
                            potential_fall = True
                            logger.debug(f"潜在漏检 - 置信度: {result['confidence']:.2f}")
                        
                        # 如果满足多个条件，视为潜在漏检
                        if potential_fall:
                            logger.debug(f"综合漏检判断 - 人体ID: {person_id}, 高宽比: {height_width_ratio:.2f}, 髋膝角: {hip_knee_angle:.2f}°, 肩髋角: {shoulder_hip_angle:.2f}°, 置信度: {result['confidence']:.2f}")
                            self.evaluation_stats['false_negative'] += 1
                            
                            # 调整综合评分和阈值以提高检测率
                            if result['confidence'] > 0.55:
                                # 对于接近阈值的情况，降低阈值以减少漏检
                                adjusted_result = result.copy()
                                adjusted_result['confidence'] = result['confidence'] + 0.05  # 增加一点置信度
                                adjusted_result['is_fall'] = adjusted_result['confidence'] > 0.60  # 降低阈值
                                
                                if adjusted_result['is_fall']:
                                    logger.info(f"调整检测结果为摔倒 - 原始置信度: {result['confidence']:.2f}, 调整后: {adjusted_result['confidence']:.2f}")
                                    analysis_result['analysis'] = adjusted_result
                                    
                                    # 触发报警
                                    self.trigger_alarm()
                                    
                                    # 记录调整后的摔倒事件
                                    fall_event = {
                                        'timestamp': current_time,
                                        'person_id': person_id,
                                        'confidence': adjusted_result['confidence'],
                                        'features': adjusted_result['features'],
                                        'adjusted': True
                                    }
                                    self.fall_events.append(fall_event)
            
            except Exception as e:
                logger.error(f"分析单个姿态时出错: {str(e)}")
                # 即使出错也继续处理其他检测结果
                continue
        
        # 每10次预测尝试保存统计数据
        if self.evaluation_stats['total_predictions'] % 10 == 0:
            if self.detector and hasattr(self.detector, 'save_accuracy_stats'):
                try:
                    self.detector.save_accuracy_stats()
                except Exception as e:
                    logger.error(f"保存准确率统计时出错: {str(e)}")
            
            # 记录评估统计到日志
            accuracy_rate = self.evaluation_stats['correct_predictions'] / self.evaluation_stats['total_predictions'] \
                            if self.evaluation_stats['total_predictions'] > 0 else 0
            logger.info(f"当前评估统计: 总预测数={self.evaluation_stats['total_predictions']}, "
                      f"准确率估计={accuracy_rate:.2%}")

        return analysis_results
        
    def register_detector(self, detector):
        """注册检测器实例，用于更新统计信息
        
        Args:
            detector: YoloPoseDetector实例
        """
        self.detector = detector
        logger.info("检测器已注册到摔倒分析器")
        
    def update_evaluation_stats(self, is_fall_detected, confidence):
        """更新评估统计数据
        
        Args:
            is_fall_detected: 是否检测到摔倒
            confidence: 检测置信度
        """
        # 更新内部评估统计
        self.evaluation_stats['total_predictions'] += 1
        if is_fall_detected and confidence > 0.7:  # 高置信度的检测作为正确预测
            self.evaluation_stats['correct_predictions'] += 1
        
        # 尝试更新检测器的准确率数据
        if self.detector:
            try:
                # 基于置信度估计真阳性/假阳性
                if is_fall_detected:
                    if confidence > 0.7:  # 高置信度的摔倒检测视为可能的真阳性
                        self.evaluation_stats['true_positive'] += 1
                        self.detector.log_accuracy_data(
                            self.evaluation_stats['true_positive'],
                            self.evaluation_stats['false_positive'],
                            self.evaluation_stats['false_negative']
                        )
                    else:  # 低置信度的摔倒检测视为可能的假阳性
                        self.evaluation_stats['false_positive'] += 1
                        self.detector.log_accuracy_data(
                            self.evaluation_stats['true_positive'],
                            self.evaluation_stats['false_positive'],
                            self.evaluation_stats['false_negative']
                        )
                else:
                    # 未检测到摔倒但姿态异常可能是假阴性
                    # 此处可以根据其他特征进一步判断
                    pass
                    
            except Exception as e:
                logger.error(f"更新准确率数据时出错: {str(e)}")
        
        # 每10次预测尝试保存统计数据
        if self.evaluation_stats['total_predictions'] % 10 == 0:
            if self.detector and hasattr(self.detector, 'save_accuracy_stats'):
                try:
                    self.detector.save_accuracy_stats()
                except Exception as e:
                    logger.error(f"保存准确率统计时出错: {str(e)}")
            
            # 记录评估统计到日志
            accuracy_rate = self.evaluation_stats['correct_predictions'] / self.evaluation_stats['total_predictions'] \
                            if self.evaluation_stats['total_predictions'] > 0 else 0
            logger.info(f"当前评估统计: 总预测数={self.evaluation_stats['total_predictions']}, "
                      f"准确率估计={accuracy_rate:.2%}, "
                      f"TP={self.evaluation_stats['true_positive']}, "
                      f"FP={self.evaluation_stats['false_positive']}, "
                      f"FN={self.evaluation_stats['false_negative']}")

    def update_thresholds(self, thresholds):
        """更新角度阈值

        Args:
            thresholds: 新的阈值字典
        """
        for key, value in thresholds.items():
            if key in self.angle_thresholds:
                self.angle_thresholds[key] = value
                logger.info(f"更新阈值 {key}: {value}")

    def get_fall_statistics(self):
        """获取摔倒统计信息

        Returns:
            stats: 统计信息
        """
        stats = {
            'total_falls': len(self.fall_events),
            'last_fall_time': self.fall_events[-1]['timestamp'] if self.fall_events else None,
            'recent_falls': len([event for event in self.fall_events
                                 if (datetime.now() - event['timestamp']).seconds < 3600])  # 最近1小时内
        }
        return stats

    def clear_history(self):
        """清除历史记录"""
        self.fall_history = []
        self.fall_events = []
        logger.info("历史记录已清除")