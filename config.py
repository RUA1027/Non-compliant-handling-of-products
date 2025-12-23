# -*- coding: utf-8 -*-
"""
生产线违规取放检测系统 - 配置文件
Configuration file for Production Line Violation Detection System
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "视频素材")
NORMAL_VIDEO_DIR = os.path.join(VIDEO_DIR, "正常")
ABNORMAL_VIDEO_DIR = os.path.join(VIDEO_DIR, "异常")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

# 创建必要的目录
for dir_path in [LOG_DIR, SCREENSHOT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ==================== 检测参数配置 ====================
class DetectionConfig:
    """检测算法相关配置"""
    
    # MediaPipe手部检测配置
    HAND_DETECTION_CONFIDENCE = 0.4  # 降低阈值以提高灵敏度，特别是戴手套的情况
    HAND_TRACKING_CONFIDENCE = 0.4   # 降低跟踪阈值
    MAX_NUM_HANDS = 4                # 最大检测手部数量
    
    # 肤色检测配置 (YCrCb色彩空间)
    SKIN_YCRCB_MIN = (0, 133, 77)    # YCrCb下界
    SKIN_YCRCB_MAX = (255, 173, 127) # YCrCb上界
    SKIN_AREA_THRESHOLD = 500        # 肤色区域面积阈值
    
    # 运动检测配置
    MOTION_THRESHOLD = 25            # 帧差阈值
    MOTION_AREA_THRESHOLD = 1000     # 运动区域面积阈值
    
    # ROI区域配置 (相对坐标 0-1)
    # 设置为None表示检测整个画面
    ROI_ENABLED = False              # 是否启用ROI
    ROI_X = 0.1                     # ROI起始x (相对)
    ROI_Y = 0.1                      # ROI起始y (相对)
    ROI_WIDTH = 0.8                  # ROI宽度 (相对)
    ROI_HEIGHT = 0.8                 # ROI高度 (相对)

# ==================== 报警参数配置 ====================
class AlarmConfig:
    """报警系统相关配置"""
    
    # 报警触发配置
    ALARM_FRAME_THRESHOLD = 8        # 降低连续帧数阈值，提高报警响应速度
    ALARM_COOLDOWN_SECONDS = 3       # 恢复较短的冷却时间
    
    # 报警级别
    LEVEL_NORMAL = 0                 # 正常
    LEVEL_WARNING = 1                # 警告（检测到可疑情况）
    LEVEL_DANGER = 2                 # 危险（确认违规操作）
    
    # 视觉报警配置
    NORMAL_COLOR = (0, 255, 0)       # 正常状态颜色 (BGR: 绿色)
    WARNING_COLOR = (0, 255, 255)    # 警告状态颜色 (BGR: 黄色)
    DANGER_COLOR = (0, 0, 255)       # 危险状态颜色 (BGR: 红色)
    
    # 声音报警配置
    SOUND_ENABLED = True             # 是否启用声音报警
    ALARM_FREQUENCY = 1000           # 报警声频率 (Hz)
    ALARM_DURATION = 500             # 报警声持续时间 (ms)

# ==================== 显示参数配置 ====================
class DisplayConfig:
    """显示相关配置"""
    
    # 窗口配置
    WINDOW_WIDTH = 1280              # 窗口宽度
    WINDOW_HEIGHT = 720              # 窗口高度
    
    # 显示元素
    SHOW_FPS = True                  # 显示帧率
    SHOW_DETECTION_BOX = True        # 显示检测框
    SHOW_HAND_LANDMARKS = True       # 显示手部关键点
    SHOW_STATUS_BAR = True           # 显示状态栏
    
    # 字体配置
    FONT_SCALE = 0.8                 # 字体大小
    FONT_THICKNESS = 2               # 字体粗细

# ==================== 视频处理配置 ====================
class VideoConfig:
    """视频处理相关配置"""
    
    # 处理配置
    PROCESS_SCALE = 1.0              # 处理缩放比例（降低可提高性能）
    FRAME_SKIP = 0                   # 跳帧数（0表示不跳帧）
    
    # 录制配置
    RECORD_ENABLED = False           # 是否录制
    RECORD_FPS = 25                  # 录制帧率
    RECORD_CODEC = 'mp4v'            # 视频编码器

# ==================== 日志配置 ====================
class LogConfig:
    """日志相关配置"""
    
    LOG_LEVEL = "INFO"               # 日志级别
    LOG_TO_FILE = True               # 是否写入文件
    LOG_TO_CONSOLE = True            # 是否输出到控制台
    MAX_LOG_FILES = 30               # 最大日志文件数
