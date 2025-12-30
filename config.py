# -*- coding: utf-8 -*-
"""
生产线违规取放检测系统 - 配置文件
Configuration file for Production Line Violation Detection System
"""

import os

# ==================== 路径配置 ====================
# 使用绝对路径确保在不同环境下运行都能正确找到资源
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "视频素材")
NORMAL_VIDEO_DIR = os.path.join(VIDEO_DIR, "正常")
ABNORMAL_VIDEO_DIR = os.path.join(VIDEO_DIR, "异常")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

# 自动初始化项目结构：确保日志和截图目录存在，防止运行时IO报错
for dir_path in [LOG_DIR, SCREENSHOT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ==================== 检测参数配置 ====================
class DetectionConfig:
    """
    检测算法核心参数：
    包含 MediaPipe 手部检测、YCrCb 肤色模型以及运动检测的阈值设定。
    """
    
    # MediaPipe 手部检测配置
    # 灵敏度权衡：较低的阈值(0.35)能更好地捕捉戴手套或光照不足时的手部，但可能增加误报
    HAND_DETECTION_CONFIDENCE = 0.4  
    HAND_TRACKING_CONFIDENCE = 0.4   
    MAX_NUM_HANDS = 5                # 产线场景下可能出现多只手同时操作
    
    # 肤色检测配置 (基于 YCrCb 色彩空间)
    # 相比 RGB，YCrCb 能更好地分离亮度(Y)和色度(Cr, Cb)，对光照变化更鲁棒
    SKIN_YCRCB_MIN = (0, 133, 77)    
    SKIN_YCRCB_MAX = (255, 173, 127) 
    SKIN_AREA_THRESHOLD = 500        # 过滤掉画面中细小的肤色噪点
    
    # 运动检测配置 (基于帧差法)
    MOTION_THRESHOLD = 30            # 像素值变化阈值，用于判定该点是否在运动（提高以减少反光误报）
    MOTION_AREA_THRESHOLD = 1500     # 运动连通域面积阈值，过滤机械振动等微小干扰（提高以过滤小面积噪声）
    
    # ROI (Region of Interest) 感兴趣区域配置
    # 核心优化：通过限定检测区域，排除产线边缘机械臂、背景行人等无关干扰
    ROI_ENABLED = True               
    ROI_X = 0.12                      # 区域起始横坐标 (比例)
    ROI_Y = 0.07                     # 区域起始纵坐标 (比例)
    ROI_WIDTH = 0.83                  # 区域宽度 (比例)
    ROI_HEIGHT = 0.88                 # 区域高度 (比例)

# ==================== 报警参数配置 ====================
class AlarmConfig:
    """
    报警逻辑配置：
    定义了从“检测到手”到“触发警报”的时间过滤逻辑。
    """
    
    # 报警触发配置
    # 时间过滤机制：只有当连续 8 帧检测到违规，才判定为真实违规，有效过滤瞬时误检
    ALARM_FRAME_THRESHOLD = 8        
    # 报警冷却：触发一次报警后，3秒内不再重复触发，避免日志冗余
    ALARM_COOLDOWN_SECONDS = 3       
    
    # 报警级别定义
    LEVEL_NORMAL = 0                 # 正常状态
    LEVEL_WARNING = 1                # 预警状态（疑似违规）
    LEVEL_DANGER = 2                 # 危险状态（确认违规）
    
    # 视觉反馈颜色 (OpenCV 使用 BGR 格式)
    NORMAL_COLOR = (0, 255, 0)       # 绿色
    WARNING_COLOR = (0, 255, 255)    # 黄色
    DANGER_COLOR = (0, 0, 255)       # 红色
    
    # 声音报警配置
    SOUND_ENABLED = True             
    ALARM_FREQUENCY = 1000           # 蜂鸣器频率 (Hz)
    ALARM_DURATION = 500             # 每次鸣叫持续时间 (ms)

# ==================== 显示参数配置 ====================
class DisplayConfig:
    """
    UI 渲染配置：
    控制 GUI 界面上显示哪些辅助信息。
    """
    
    # 窗口分辨率
    WINDOW_WIDTH = 1280              
    WINDOW_HEIGHT = 720              
    
    # 渲染开关
    SHOW_FPS = True                  # 实时帧率显示，用于评估算法性能
    SHOW_DETECTION_BOX = True        # 绘制手部包围框
    SHOW_HAND_LANDMARKS = True       # 绘制 MediaPipe 的 21 个手部关键点
    SHOW_STATUS_BAR = True           # 底部状态栏显示
    
    # 视觉样式
    FONT_SCALE = 0.8                 
    FONT_THICKNESS = 2               

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
