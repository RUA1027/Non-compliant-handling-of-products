# -*- coding: utf-8 -*-
"""
视频处理与渲染模块 (Video Processing & Rendering Module)
--------------------------------------------------
【作用与功能】
本模块是系统的“数据流与视觉层”，负责视频数据的 I/O 和最终画面的绘制。
主要功能包括：视频源读取、图像预处理（缩放/跳帧）、UI 元素叠加（中文/特效）。

【使用的工具与技术】
1. OpenCV (cv2)：VideoCapture 读取视频，VideoWriter 录制结果。
2. PIL (Pillow)：实现跨平台的中文字符绘制（解决 OpenCV 不支持中文问题）。
3. 性能优化：
   - 动态缩放 (Resize)：降低分辨率以提升检测 FPS。
   - 跳帧机制 (Frame Skip)：高负载下丢弃部分帧以保持实时性。

【实现方式】
- VideoProcessor 类：封装 OpenCV 视频流接口，计算实时 FPS。
- FrameRenderer 类：提供 draw_status_bar, draw_alarm_overlay 等绘图方法。
  使用“PIL 桥接技术”在 OpenCV 图像上绘制美观的中文字体。
--------------------------------------------------
"""

import cv2
import time
import numpy as np
from config import DisplayConfig, VideoConfig


class VideoProcessor:
    """
    视频处理引擎：
    负责视频流的生命周期管理，包括多源输入（摄像头/文件）、预处理（缩放/跳帧）以及视频录制。
    """
    
    def __init__(self, source=0):
        """
        初始化视频处理器。
        
        Args:
            source: 视频源。0 代表默认摄像头，字符串代表视频文件路径。
        """
        self.source = source
        self.cap = None
        self.is_running = False
        
        # 性能统计：用于实时计算处理帧率 (FPS)
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 视频持久化：用于将处理后的画面保存为视频文件
        self.video_writer = None
        
    def open(self):
        """打开视频源并获取元数据（分辨率、总帧数等）"""
        self.cap = cv2.VideoCapture(self.source)
        
        if self.cap is None or not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 提取视频流属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read(self):
        """
        读取并预处理一帧图像。
        包含跳帧逻辑（降低 CPU 负载）和缩放逻辑（提升检测速度）。
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None or not self.is_running:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.frame_count += 1
            
            # 性能优化：跳帧处理。在处理能力有限时，可以跳过中间帧以维持实时性
            if VideoConfig.FRAME_SKIP > 0:
                for _ in range(VideoConfig.FRAME_SKIP):
                    self.cap.read()
                    self.frame_count += 1
            
            # 性能优化：缩放处理。减小输入分辨率可显著加快深度学习模型的推理速度
            if VideoConfig.PROCESS_SCALE != 1.0:
                new_width = int(self.width * VideoConfig.PROCESS_SCALE)
                new_height = int(self.height * VideoConfig.PROCESS_SCALE)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 实时 FPS 计算：反映系统当前的实际运行效率
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
    视觉渲染引擎：
    负责在原始视频帧上叠加显示检测框、状态栏、报警特效以及中文字符。
    """
    
    def __init__(self):
        """初始化渲染器，配置中文字体支持"""
        # 跨平台中文支持：OpenCV 原生不支持中文，此处通过 PIL (Pillow) 库进行桥接
        self.use_pil = False
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.pil_available = True
            # 自动搜索系统中可用的中文字体
            font_names = ["simhei.ttf", "msyh.ttc", "STHeiti Medium.ttc", "DroidSansFallback.ttf"]
            self.font = None
            for font_name in font_names:
                try:
                    self.font = ImageFont.truetype(font_name, 24)
                    break
                except:
                    continue
            
            if self.font is None:
                self.font = ImageFont.load_default()
            else:
                self.use_pil = True
        except ImportError:
            self.pil_available = False
    
    def draw_status_bar(self, frame, status_text, status_color, fps=0.0):
        """
        在画面顶部绘制半透明状态栏。
        
        Args:
            frame: 图像帧
            status_text: 当前系统状态描述（支持中文）
            status_color: 状态指示灯颜色
            fps: 实时帧率
        """
        h, w = frame.shape[:2]
        
        # 绘制半透明背景：增强文字可读性
        bar_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 绘制状态指示灯（圆形）
        indicator_radius = 15
        cv2.circle(frame, (30, bar_height // 2), indicator_radius, status_color, -1)
        
        # 准备文本数据
        texts = []
        texts.append((status_text, (60, bar_height // 2 - 12), (255, 255, 255)))
        
        if DisplayConfig.SHOW_FPS:
            fps_text = f"FPS: {fps:.1f}"
            texts.append((fps_text, (w - 120, bar_height // 2 - 10), (200, 200, 200)))
            
        # 统一渲染：减少 OpenCV 与 PIL 之间的转换次数，优化性能
        return self._draw_texts(frame, texts)

    def _draw_texts(self, frame, texts):
        """
        文本渲染核心方法：自动识别语种并选择渲染引擎。
        """
        has_chinese = any(self._contains_chinese(t[0]) for t in texts)
        
        if self.use_pil and has_chinese:
            # 包含中文时，切换到 PIL 渲染引擎
            from PIL import Image, ImageDraw
            
            # 格式转换：OpenCV (BGR) -> PIL (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            for text, pos, color in texts:
                # PIL 使用 RGB 颜色，需反转 OpenCV 的 BGR
                draw.text(pos, text, font=self.font, fill=color[::-1])
            
            # 格式转换：PIL (RGB) -> OpenCV (BGR)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # 纯英文场景，直接使用 OpenCV 原生函数，效率更高
            for text, pos, color in texts:
                cv_pos = (pos[0], pos[1] + 20)
                cv2.putText(frame, text, cv_pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, color, 2)
            return frame

    def draw_roi(self, frame, roi, color=(255, 255, 0)):
        """绘制感兴趣区域 (ROI) 的边界框"""
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def draw_alarm_overlay(self, frame, alarm_level):
        """
        绘制全屏报警特效。
        危险级别时，画面四周会闪烁红色边框，并显示醒目的警告标语。
        """
        from config import AlarmConfig
        
        if alarm_level == AlarmConfig.LEVEL_DANGER:
            h, w = frame.shape[:2]
            border_width = 10
            
            # 闪烁逻辑：利用时间戳实现 2Hz 的视觉闪烁效果
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_width)
                
                warning_text = "!!! VIOLATION DETECTED !!!"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 50
                
                # 绘制文字背景块，增强视觉冲击力
                cv2.rectangle(frame, (text_x - 10, text_y - 40),
                             (text_x + text_size[0] + 10, text_y + 10), (0, 0, 150), -1)
                
                frame = self._draw_texts(frame, [(warning_text, (text_x, text_y - 30), (255, 255, 255))])
        
        elif alarm_level == AlarmConfig.LEVEL_WARNING:
            # 警告级别：仅显示黄色边框
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 5)
        
        return frame
    
    def draw_info_panel(self, frame, info_dict):
        """
        绘制信息面板
        """
        h, w = frame.shape[:2]
        panel_width = 250
        panel_height = len(info_dict) * 30 + 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width - 10, 70),
                     (w - 10, 70 + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        texts = []
        y_offset = 95
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            texts.append((text, (w - panel_width, y_offset - 20), (255, 255, 255)))
            y_offset += 30
            
        return self._draw_texts(frame, texts)

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
