# -*- coding: utf-8 -*-
"""
手部检测模块
Hand Detection Module using MediaPipe and Skin Color Detection
"""

import cv2
import numpy as np
import mediapipe as mp
from config import DetectionConfig

class HandDetector:
    """
    手部检测器类
    结合MediaPipe手部检测和肤色检测实现高精度检测
    """
    
    def __init__(self):
        """初始化手部检测器"""
        # MediaPipe手部检测器
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=DetectionConfig.MAX_NUM_HANDS,
            min_detection_confidence=DetectionConfig.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=DetectionConfig.HAND_TRACKING_CONFIDENCE
        )
        
        # 帧差法用的前一帧
        self.prev_frame = None
        
        # 检测结果缓存
        self.detection_history = []
        self.history_size = 5
        
    def detect_hands_mediapipe(self, frame):
        """
        使用MediaPipe检测手部
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            results: MediaPipe检测结果
            hand_boxes: 手部边界框列表 [(x, y, w, h), ...]
        """
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(rgb_frame)
        
        hand_boxes = []
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                # 计算边界框
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
                
                # 扩展边界框
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                hand_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return results, hand_boxes
    
    def detect_skin_color(self, frame):
        """
        使用肤色检测手部区域
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            mask: 肤色掩码
            contours: 肤色区域轮廓
            skin_boxes: 肤色区域边界框
        """
        # 转换为YCrCb色彩空间
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # 创建肤色掩码
        lower = np.array(DetectionConfig.SKIN_YCRCB_MIN, dtype=np.uint8)
        upper = np.array(DetectionConfig.SKIN_YCRCB_MAX, dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        skin_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > DetectionConfig.SKIN_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(cnt)
                # 过滤过小或过大的区域
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # 合理的宽高比
                    skin_boxes.append((x, y, w, h))
        
        return mask, contours, skin_boxes
    
    def detect_motion(self, frame):
        """
        使用帧差法检测运动区域
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            motion_mask: 运动掩码
            motion_boxes: 运动区域边界框
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_boxes = []
        motion_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if self.prev_frame is not None:
            # 计算帧差
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            
            # 二值化
            _, motion_mask = cv2.threshold(
                frame_diff, 
                DetectionConfig.MOTION_THRESHOLD, 
                255, 
                cv2.THRESH_BINARY
            )
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
            
            # 查找运动轮廓
            contours, _ = cv2.findContours(
                motion_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > DetectionConfig.MOTION_AREA_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(cnt)
                    motion_boxes.append((x, y, w, h))
        
        self.prev_frame = gray.copy()
        
        return motion_mask, motion_boxes
    
    def detect(self, frame, roi=None):
        """
        综合检测方法
        
        Args:
            frame: BGR格式的图像帧
            roi: 感兴趣区域 (x, y, w, h)，None表示全图
            
        Returns:
            dict: 检测结果字典
        """
        # 如果指定了ROI，裁剪图像
        if roi is not None:
            x, y, w, h = roi
            detect_frame = frame[y:y+h, x:x+w].copy()
            roi_offset = (x, y)
        else:
            detect_frame = frame
            roi_offset = (0, 0)
        
        # MediaPipe手部检测
        mp_results, mp_boxes = self.detect_hands_mediapipe(detect_frame)
        
        # 肤色检测
        skin_mask, skin_contours, skin_boxes = self.detect_skin_color(detect_frame)
        
        # 运动检测
        motion_mask, motion_boxes = self.detect_motion(detect_frame)
        
        # 调整边界框坐标（考虑ROI偏移）
        if roi_offset != (0, 0):
            ox, oy = roi_offset
            mp_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in mp_boxes]
            skin_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in skin_boxes]
            motion_boxes = [(x+ox, y+oy, w, h) for x, y, w, h in motion_boxes]
        
        # 综合判断是否检测到手部
        hand_detected = len(mp_boxes) > 0
        
        # 增强逻辑：处理戴手套的情况
        # 如果MediaPipe没有检测到，但存在显著的运动区域
        if not hand_detected and len(motion_boxes) > 0:
            for m_box in motion_boxes:
                mx, my, mw, mh = m_box
                # 检查运动物体的比例是否像手（避免细长条的干扰）
                aspect_ratio = mw / mh if mh > 0 else 0
                area = mw * mh
                
                # 如果运动区域面积足够大，且比例接近手部（0.3 - 3.0）
                if area > DetectionConfig.MOTION_AREA_THRESHOLD * 1.2 and 0.3 < aspect_ratio < 3.0:
                    # 即使没有肤色（戴手套），只要运动特征明显，也判定为检测到
                    hand_detected = True
                    break
        
        # 更新检测历史
        self.detection_history.append(hand_detected)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # 稳定检测结果（非对称时间滤波）
        # 开启报警需要多数帧确认（防误报），关闭报警只需少数帧确认（提高恢复速度）
        if hand_detected:
            # 开启：需要历史记录中超过一半的帧检测到手
            stable_detection = sum(self.detection_history) >= len(self.detection_history) // 2 + 1
        else:
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
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """
        判断两个边界框是否重叠
        
        Args:
            box1, box2: 边界框 (x, y, w, h)
            threshold: IoU阈值
            
        Returns:
            bool: 是否重叠
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi1 >= xi2 or yi1 >= yi2:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
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
