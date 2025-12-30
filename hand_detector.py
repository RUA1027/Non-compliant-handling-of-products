# -*- coding: utf-8 -*-
"""
手部检测核心模块 (Hand Detection Module)
--------------------------------------------------
【作用与功能】
本模块是系统的“视觉引擎”，负责从图像中精准识别手部目标。
主要功能包括：多模态手部检测、干扰过滤、运动轨迹分析。

【使用的工具与技术】
1. MediaPipe Hands：Google 开源的轻量级手部关键点检测模型（深度学习）。
2. 传统机器视觉 (OpenCV)：
   - YCrCb 肤色分割：利用色度空间特性提取裸手区域。
   - 帧差法 (Frame Difference)：提取运动前景，辅助检测戴手套的手。
   - 形态学操作：开/闭运算去噪。
3. 几何分析：基于轮廓面积、长宽比过滤非手部物体。

【实现方式】
- HandDetector 类：实现“多模态融合检测”策略。
  1. 优先使用 MediaPipe 进行关键点回归。
  2. 若 MediaPipe 失效，回退到“运动+肤色”联合检测。
  3. 引入 _is_periodic_motion 算法，通过历史轨迹分析过滤齿轮等周期性干扰。
  4. 采用时域滤波 (Temporal Filtering) 稳定检测结果。
--------------------------------------------------
"""

import cv2
import numpy as np
import mediapipe as mp
from config import DetectionConfig

class HandDetector:
    """
    多模态手部检测器：
    融合了深度学习 (MediaPipe)、传统色彩模型 (YCrCb 肤色检测) 以及运动分析 (帧差法)。
    旨在解决产线复杂背景、光照变化以及工人戴手套操作等极端情况下的检测难题。
    """
    
    def __init__(self):
        """初始化检测引擎，配置 MediaPipe 及相关历史缓冲区"""
        # MediaPipe 官方手部模型初始化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # 设置为 False 以利用视频流的帧间关联性
            max_num_hands=DetectionConfig.MAX_NUM_HANDS,
            min_detection_confidence=DetectionConfig.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=DetectionConfig.HAND_TRACKING_CONFIDENCE
        )
        
        # 运动检测：存储前一帧灰度图，用于计算帧间差异
        self.prev_frame = None
        
        # 结果平滑：存储最近几帧的检测状态，通过时间域投票减少闪烁
        self.detection_history = []
        self.history_size = 5
        
        # 周期性干扰过滤：记录运动物体的中心点轨迹，用于识别并排除齿轮等重复运动物体
        self.motion_position_history = []
        self.motion_history_size = 15
        
    def detect_hands_mediapipe(self, frame):
        """
        基于深度学习模型检测手部关键点。
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            results: 包含 21 个手部关键点的原始数据
            hand_boxes: 转换后的手部边界框列表 [(x, y, w, h), ...]
        """
        # MediaPipe 要求输入为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部并执行推理
        results = self.hands.process(rgb_frame)
        
        hand_boxes = []
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                # 遍历 21 个关键点，计算其在图像中的像素坐标范围
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
                
                # 适当扩充边界框，确保覆盖整个手掌边缘
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                hand_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return results, hand_boxes
    
    def detect_skin_color(self, frame):
        """
        基于 YCrCb 色彩空间的肤色区域分割。
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            mask: 二值化肤色掩码
            contours: 提取的区域轮廓
            skin_boxes: 过滤后的肤色区域边界框
        """
        # 转换色彩空间：YCrCb 对光照变化（Y通道）具有更好的鲁棒性
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # 阈值分割：提取符合人类肤色特征的 Cr 和 Cb 通道范围
        lower = np.array(DetectionConfig.SKIN_YCRCB_MIN, dtype=np.uint8)
        upper = np.array(DetectionConfig.SKIN_YCRCB_MAX, dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # 形态学处理：开运算去除噪点，闭运算填补空洞，膨胀连接断开区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 轮廓分析：过滤掉面积过小或宽高比不合理的非手部区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        skin_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > DetectionConfig.SKIN_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                # 过滤掉过于细长（如电线）或过于扁平的干扰物
                if 0.2 < aspect_ratio < 5.0:
                    skin_boxes.append((x, y, w, h))
        
        return mask, contours, skin_boxes
    
    def detect_motion(self, frame):
        """
        基于帧差法的运动目标检测。
        用于辅助识别戴手套的手（此时肤色检测失效，MediaPipe 效果下降）。
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            motion_mask: 运动区域掩码
            motion_boxes: 运动目标边界框
        """
        # 预处理：灰度化并进行高斯模糊，减少摄像头传感器噪声的影响
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_boxes = []
        motion_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if self.prev_frame is not None:
            # 计算当前帧与前一帧的绝对差值
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            
            # 二值化：将显著变化的像素标记为运动点
            _, motion_mask = cv2.threshold(
                frame_diff, 
                DetectionConfig.MOTION_THRESHOLD, 
                255, 
                cv2.THRESH_BINARY
            )
            
            # 形态学优化：消除孤立噪点并合并相邻运动块
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
            
            # 提取运动轮廓
            contours, _ = cv2.findContours(
                motion_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # 面积过滤：排除微小的背景抖动
                if area > DetectionConfig.MOTION_AREA_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / h if h > 0 else 0
                    # 形状过滤：排除过于细长的机械臂运动
                    if 0.25 < aspect_ratio < 4.0:
                        motion_boxes.append((x, y, w, h))
        
        self.prev_frame = gray.copy()
        
        return motion_mask, motion_boxes
    
    def detect(self, frame, roi=None):
        """
        综合检测逻辑：融合多种算法结果并进行时域滤波。
        
        Args:
            frame: 原始图像帧
            roi: 感兴趣区域 (x, y, w, h)
            
        Returns:
            dict: 包含所有检测细节的结果字典
        """
        # ROI 裁剪：只关注作业核心区域，提升处理速度并减少背景干扰
        if roi is not None:
            x, y, w, h = roi
            detect_frame = frame[y:y+h, x:x+w].copy()
            roi_offset = (x, y)
        else:
            detect_frame = frame
            roi_offset = (0, 0)
        
        # 1. 深度学习检测 (MediaPipe)
        mp_results, mp_boxes = self.detect_hands_mediapipe(detect_frame)
        
        # 2. 传统视觉检测 (肤色 + 运动)
        skin_mask, skin_contours, skin_boxes = self.detect_skin_color(detect_frame)
        motion_mask, motion_boxes = self.detect_motion(detect_frame)
        
        # 坐标还原：将 ROI 局部坐标映射回原始图像坐标系
        if roi_offset != (0, 0):
            ox, oy = roi_offset
            mp_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in mp_boxes]
            skin_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in skin_boxes]
            motion_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in motion_boxes]
        
        # 3. 智能过滤逻辑：排除产线上的固定干扰物
        # 产线上的细小零件、反光点或远处的干扰物虽然可能被模型误识别，但其面积往往较小。
        valid_mp_boxes = []
        for box in mp_boxes:
            bx, by, bw, bh = box
            center = (bx + bw // 2, by + bh // 2)
            area = bw * bh
            
            # 尺寸过滤：排除过小的误报点
            if area < 2500: 
                continue
                
            # 周期性过滤：如果某个位置长期有“手”且不怎么移动，判定为机械结构（如传送带上的固定标识）
            if self._is_periodic_motion(center, radius_threshold=30, occurrence_threshold=0.8):
                continue
                
            valid_mp_boxes.append(box)
        
        mp_boxes = valid_mp_boxes
        
        # 4. 综合判定：MediaPipe 是主要依据
        hand_detected = len(mp_boxes) > 0
        
        # 更新运动轨迹历史，用于后续帧的周期性判定
        current_centers = []
        for box in mp_boxes:
            bx, by, bw, bh = box
            current_centers.append((bx + bw // 2, by + bh // 2))
        for m_box in motion_boxes:
            mx, my, mw, mh = m_box
            current_centers.append((mx + mw // 2, my + mh // 2))
        
        self.motion_position_history.append(current_centers)
        if len(self.motion_position_history) > self.motion_history_size:
            self.motion_position_history.pop(0)
        
        # 5. 增强逻辑：处理戴手套等 MediaPipe 失效的情况
        # 如果 MediaPipe 没检出，但存在显著且符合手部特征的运动区域，则补报
        if not hand_detected and len(motion_boxes) > 0:
            for m_box in motion_boxes:
                mx, my, mw, mh = m_box
                center = (mx + mw // 2, my + mh // 2)
                aspect_ratio = mw / mh if mh > 0 else 0
                area = mw * mh
                
                if self._is_periodic_motion(center):
                    continue
                
                # 戴手套检测：面积较大且比例符合手掌特征
                if area > DetectionConfig.MOTION_AREA_THRESHOLD * 1.5 and 0.35 < aspect_ratio < 2.8:
                    hand_detected = True
                    break
        
        # 6. 时域滤波 (Temporal Filtering)：
        # 采用非对称策略：开启报警需多帧确认（防抖），关闭报警需快速响应（灵敏）
        self.detection_history.append(hand_detected)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        if hand_detected:
            # 开启判定：历史记录中超过半数帧检测到手
            stable_detection = sum(self.detection_history) >= len(self.detection_history) // 2 + 1
        else:
            # 快速恢复：如果最近 2 帧都没检测到，立即判定为手部
            # 关闭：如果当前帧没检测到，且历史记录中没检测到的比例较高，则立即关闭
            # 这里设置为：只要最近 2 帧都没检测到，就立即判定为离开
            if len(self.detection_history) >= 2 and not self.detection_history[-1] and not self.detection_history[-2]:
                stable_detection = False
            else:
                stable_detection = sum(self.detection_history) >= len(self.detection_history) // 2 + 1
        
        result = {
            'hand_detected': hand_detected,
            'stable_detection': stable_detection,
            'mediapipe_results': mp_results,
            'mediapipe_boxes': mp_boxes,
            'skin_mask': skin_mask,
            'skin_boxes': skin_boxes,
            'motion_mask': motion_mask,
            'motion_boxes': motion_boxes,
            'detection_confidence': sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
        }
        
        return result
    
    def _is_periodic_motion(self, center, radius_threshold=50, occurrence_threshold=0.6):
        """
        判断运动是否为周期性运动（如齿轮）
        通过检测该位置是否在历史记录中反复出现来判断
        当新一帧检测到目标时，计算其中心点与历史记录中所有点的欧式距离。
        如果该目标在过去 15 帧中，有超过 80% 的时间都出现在同一个半径范围内
        系统就判定它是一个“固定干扰源”而非人手。

        Args:
            center: 运动区域中心 (x, y)
            radius_threshold: 位置匹配半径阈值
            occurrence_threshold: 出现频率阈值
            
        Returns:
            bool: 是否为周期性运动
        """
        if len(self.motion_position_history) < 10:
            return False
        
        occurrence_count = 0
        for frame_centers in self.motion_position_history:
            for hist_center in frame_centers:
                dist = ((center[0] - hist_center[0])**2 + (center[1] - hist_center[1])**2)**0.5
                if dist < radius_threshold:
                    occurrence_count += 1
                    break
        
        # 如果该位置在大部分历史帧中都出现过，认为是周期性运动
        return occurrence_count / len(self.motion_position_history) > occurrence_threshold
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """
        计算两个边界框的交并比 (IoU)，判断其重叠程度，实现多源信息互证。
        系统同时运行了多种算法，有着来自不同算法的检测边界框，它们可能会在同一个位置都检测到目标。

        该函数常用于多模态结果融合或去重，以判定不同算法检测到的是否为同一目标。
        通过计算 IoU来量化不同算法检测结果的重叠程度。
        这不仅能帮助合并重复的检测目标，还能在复杂的产线背景下，通过空间约束来过滤掉那些位置孤立、不符合逻辑的误报区域。

        算法原理：
        IoU = (Box1 ∩ Box2) / (Box1 ∪ Box2)
        
        Args:
            box1, box2: 边界框 (x, y, w, h)
            threshold: IoU 阈值，超过此值判定为重叠
            
        Returns:
            bool: 是否重叠
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 1. 计算交集区域 (Intersection) 的坐标
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        # 如果交集区域的宽或高小于等于 0，说明完全不相交
        if xi1 >= xi2 or yi1 >= yi2:
            return False
        
        # 2. 计算交集面积
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # 3. 计算并集面积 (Union)
        # 并集面积 = 矩形A面积 + 矩形B面积 - 交集面积
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # 4. 计算 IoU 值
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold
    
    def draw_results(self, frame, detection_result, show_landmarks=True, show_boxes=True):
        """
        在图像上绘制检测结果
        
        Args:
            frame: 原始图像帧
            detection_result: detect()方法返回的结果字典
            show_landmarks: 是否显示手部关键点
            show_boxes: 是否显示边界框
            
        Returns:
            frame: 绘制后的图像帧
        """
        output = frame.copy()
        
        # 绘制MediaPipe手部关键点
        if show_landmarks and detection_result['mediapipe_results'].multi_hand_landmarks:
            for hand_landmarks in detection_result['mediapipe_results'].multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # 绘制手部边界框
        if show_boxes:
            for box in detection_result['mediapipe_boxes']:
                x, y, w, h = box
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output, "HAND", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output
    
    def release(self):
        """释放资源"""
        self.hands.close()


class SkinDetector:
    """
    独立的肤色检测器类
    用于在MediaPipe不可用时作为备选方案
    """
    
    def __init__(self):
        """初始化肤色检测器"""
        # 使用多种色彩空间的肤色模型
        self.color_models = {
            'YCrCb': {
                'lower': np.array([0, 133, 77], dtype=np.uint8),
                'upper': np.array([255, 173, 127], dtype=np.uint8)
            },
            'HSV': {
                'lower': np.array([0, 20, 70], dtype=np.uint8),
                'upper': np.array([20, 255, 255], dtype=np.uint8)
            }
        }
    
    def detect(self, frame, method='YCrCb'):
        """
        检测肤色区域
        
        Args:
            frame: BGR格式的图像
            method: 使用的色彩空间方法
            
        Returns:
            mask: 肤色掩码
            boxes: 肤色区域边界框列表
        """
        if method == 'YCrCb':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        elif method == 'HSV':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        model = self.color_models[method]
        mask = cv2.inRange(converted, model['lower'], model['upper'])
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
        
        return mask, boxes
