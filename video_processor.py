# -*- coding: utf-8 -*-
"""
视频处理模块
Video Processing Module for Production Line Violation Detection
"""

import cv2
import time
import numpy as np
from config import DisplayConfig, VideoConfig


class VideoProcessor:
    """
    视频处理器类
    负责视频的读取、处理、显示
    """
    
    def __init__(self, source=0):
        """
        初始化视频处理器
        
        Args:
            source: 视频源，可以是摄像头索引(int)或视频文件路径(str)
        """
        self.source = source
        self.cap = None
        self.is_running = False
        
        # 帧率计算
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 录制器
        self.video_writer = None
        
    def open(self):
        """打开视频源"""
        self.cap = cv2.VideoCapture(self.source)
        
        if self.cap is None or not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 获取视频信息
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read(self):
        """
        读取一帧
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None or not self.is_running:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.frame_count += 1
            
            # 跳帧处理
            if VideoConfig.FRAME_SKIP > 0:
                for _ in range(VideoConfig.FRAME_SKIP):
                    self.cap.read()
                    self.frame_count += 1
            
            # 缩放处理
            if VideoConfig.PROCESS_SCALE != 1.0:
                new_width = int(self.width * VideoConfig.PROCESS_SCALE)
                new_height = int(self.height * VideoConfig.PROCESS_SCALE)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 计算帧率
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = float(self.frame_count / elapsed)
        
        return ret, frame
    
    def get_progress(self):
        """获取视频播放进度"""
        if self.cap is not None and self.total_frames > 0:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return current_frame / self.total_frames
        return 0.0
    
    def seek(self, position):
        """
        跳转到指定位置
        
        Args:
            position: 位置比例 (0-1)
        """
        if self.cap is not None and self.total_frames > 0:
            frame_num = int(position * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    def start_recording(self, output_path):
        """
        开始录制视频
        
        Args:
            output_path: 输出文件路径
        """
        fourcc = cv2.VideoWriter_fourcc(*VideoConfig.RECORD_CODEC)
        if self.width > 0 and self.height > 0:
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                VideoConfig.RECORD_FPS,
                (self.width, self.height)
            )
    
    def write_frame(self, frame):
        """写入一帧到录制文件"""
        if self.video_writer is not None and frame is not None:
            self.video_writer.write(frame)
    
    def stop_recording(self):
        """停止录制"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
    
    def release(self):
        """释放资源"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.stop_recording()


class FrameRenderer:
    """
    帧渲染器类
    负责在图像上绘制各种信息
    """
    
    def __init__(self):
        """初始化渲染器"""
        # 尝试加载中文字体（如果PIL可用）
        self.use_pil = False
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.pil_available = True
            # 尝试加载中文字体
            try:
                self.font = ImageFont.truetype("simhei.ttf", 24)
                self.use_pil = True
            except:
                self.font = ImageFont.load_default()
        except ImportError:
            self.pil_available = False
    
    def draw_status_bar(self, frame, status_text, status_color, fps=0.0):
        """
        绘制状态栏
        
        Args:
            frame: 图像帧
            status_text: 状态文本
            status_color: 状态颜色 (BGR)
            fps: 帧率
            
        Returns:
            frame: 绘制后的图像帧
        """
        h, w = frame.shape[:2]
        
        # 绘制状态栏背景
        bar_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 绘制状态指示器
        indicator_radius = 15
        cv2.circle(frame, (30, bar_height // 2), indicator_radius, status_color, -1)
        
        # 绘制状态文本
        self._put_text(frame, status_text, (60, bar_height // 2 + 8),
                      color=(255, 255, 255), font_scale=0.7)
        
        # 绘制帧率
        if DisplayConfig.SHOW_FPS:
            fps_text = f"FPS: {fps:.1f}"
            self._put_text(frame, fps_text, (w - 120, bar_height // 2 + 8),
                          color=(200, 200, 200), font_scale=0.6)
        
        return frame
    
    def draw_roi(self, frame, roi, color=(255, 255, 0)):
        """
        绘制ROI区域
        
        Args:
            frame: 图像帧
            roi: ROI区域 (x, y, w, h)
            color: 边框颜色
            
        Returns:
            frame: 绘制后的图像帧
        """
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def draw_alarm_overlay(self, frame, alarm_level):
        """
        绘制报警覆盖层
        
        Args:
            frame: 图像帧
            alarm_level: 报警级别
            
        Returns:
            frame: 绘制后的图像帧
        """
        from config import AlarmConfig
        
        if alarm_level == AlarmConfig.LEVEL_DANGER:
            # 红色闪烁边框
            h, w = frame.shape[:2]
            border_width = 10
            
            # 使用时间产生闪烁效果
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_width)
                
                # 绘制警告文本
                warning_text = "!!! VIOLATION DETECTED !!!"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 50
                
                # 文本背景
                cv2.rectangle(frame, (text_x - 10, text_y - 40),
                             (text_x + text_size[0] + 10, text_y + 10), (0, 0, 150), -1)
                cv2.putText(frame, warning_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        elif alarm_level == AlarmConfig.LEVEL_WARNING:
            # 黄色边框
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 5)
        
        return frame
    
    def draw_info_panel(self, frame, info_dict):
        """
        绘制信息面板
        
        Args:
            frame: 图像帧
            info_dict: 信息字典
            
        Returns:
            frame: 绘制后的图像帧
        """
        h, w = frame.shape[:2]
        panel_width = 250
        panel_height = len(info_dict) * 30 + 20
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width - 10, 70),
                     (w - 10, 70 + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 绘制信息
        y_offset = 95
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (w - panel_width, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        return frame
    
    def _put_text(self, frame, text, position, color=(255, 255, 255),
                  font_scale=0.7, thickness=2):
        """
        在图像上绘制文本（支持中文）
        
        Args:
            frame: 图像帧
            text: 文本内容
            position: 位置 (x, y)
            color: 文本颜色 (BGR)
            font_scale: 字体大小
            thickness: 字体粗细
        """
        if self.use_pil and self._contains_chinese(text):
            # 使用PIL绘制中文
            from PIL import Image, ImageDraw, ImageFont
            
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text(position, text, font=self.font, fill=color[::-1])
            frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # 使用OpenCV绘制
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness)
    
    def _contains_chinese(self, text):
        """检查文本是否包含中文"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False


def resize_with_aspect_ratio(image, width=None, height=None):
    """
    按比例缩放图像
    
    Args:
        image: 输入图像
        width: 目标宽度
        height: 目标高度
        
    Returns:
        resized: 缩放后的图像
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        new_size = (int(w * ratio), height)
    else:
        ratio = width / w
        new_size = (width, int(h * ratio))
    
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
