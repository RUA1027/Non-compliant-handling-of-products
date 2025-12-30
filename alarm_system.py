# -*- coding: utf-8 -*-
"""
报警系统模块
Alarm System Module for Production Line Violation Detection
"""

import os
import time
import threading
import logging
from datetime import datetime
from config import AlarmConfig, LogConfig, SCREENSHOT_DIR, LOG_DIR

# 设置日志
logging.basicConfig(
    level=getattr(logging, LogConfig.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"alarm_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        ) if LogConfig.LOG_TO_FILE else logging.NullHandler(),
        logging.StreamHandler() if LogConfig.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlarmSystem:
    """
    报警系统类：负责管理报警状态机、触发多媒体报警及记录违规事件。
    该类实现了从“检测结果”到“业务响应”的逻辑转换。
    """
    
    def __init__(self):
        """初始化报警系统，设置初始状态和资源"""
        self.current_level = AlarmConfig.LEVEL_NORMAL  # 当前报警级别 (0:正常, 1:警告, 2:危险)
        self.violation_count = 0                       # 连续违规帧数计数器，用于过滤噪声
        self.last_alarm_time = 0                       # 上次触发报警的时间戳（用于冷却机制）
        self.is_alarming = False                       # 当前是否正处于报警状态
        
        # 事件记录：存储最近发生的违规事件对象
        self.events = []
        self.max_events = 100                          # 内存中保留的最大事件数
        
        # 报警回调函数列表：当触发报警时，会依次调用这些函数（如通知GUI更新）
        self.alarm_callbacks = []
        
        # 声音报警线程管理：确保声音播放不阻塞主视觉处理线程
        self.sound_thread = None
        self.stop_sound = False
        
        logger.info("报警系统初始化完成")
    
    def update(self, hand_detected, confidence=0.0):
        """
        核心状态更新方法：根据当前帧的检测结果更新报警级别。
        
        Args:
            hand_detected (bool): 当前帧是否检测到手部
            confidence (float): 检测到的置信度分数
            
        Returns:
            int: 更新后的报警级别 (LEVEL_NORMAL/WARNING/DANGER)
        """
        current_time = time.time()
        
        if hand_detected:
            self.violation_count += 1
            
            # 逻辑：根据连续检测到的帧数，阶梯式提升报警级别
            if self.violation_count >= AlarmConfig.ALARM_FRAME_THRESHOLD:
                # 达到危险阈值，检查是否已过冷却期，避免频繁骚扰
                if current_time - self.last_alarm_time >= AlarmConfig.ALARM_COOLDOWN_SECONDS:
                    self.current_level = AlarmConfig.LEVEL_DANGER
                    self._trigger_alarm(confidence)
                    self.last_alarm_time = current_time
            elif self.violation_count >= AlarmConfig.ALARM_FRAME_THRESHOLD // 2:
                # 达到一半阈值，先进入警告状态
                self.current_level = AlarmConfig.LEVEL_WARNING
        else:
            # 关键逻辑：一旦手部离开，立即重置计数，实现“撤销及时”，体现系统灵敏度
            self.violation_count = 0
            self.current_level = AlarmConfig.LEVEL_NORMAL
            self.is_alarming = False
        
        return self.current_level
    
    def _trigger_alarm(self, confidence):
        """
        内部方法：触发报警动作，包括记录事件、执行回调和播放声音。
        
        Args:
            confidence (float): 触发时的检测置信度，用于记录证据
        """
        self.is_alarming = True
        
        # 构造事件字典，包含时间、级别、置信度等关键信息
        event = {
            'time': datetime.now(),
            'level': 'DANGER',
            'confidence': confidence,
            'message': '检测到违规取放操作!'
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0) # 保持内存占用稳定
        
        logger.warning(f"⚠️ 违规警报: 检测到手部操作! 置信度: {confidence:.2f}")
        
        # 执行所有注册的回调（例如：让GUI弹出红框、让日志类写入文件）
        for callback in self.alarm_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"报警回调执行失败: {e}")
        
        # 触发声音报警（如果配置开启）
        if AlarmConfig.SOUND_ENABLED:
            self._play_alarm_sound()
    
    def _play_alarm_sound(self):
        """启动独立线程播放报警音，避免阻塞主视觉处理流程"""
        if self.sound_thread is not None and self.sound_thread.is_alive():
            return # 如果已经在响了，就不重复启动
        
        self.stop_sound = False
        self.sound_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self.sound_thread.start()
    
    def _sound_worker(self):
        """声音播放工作线程：根据操作系统调用不同的音频接口"""
        try:
            import winsound
            # 循环播放指定频率和时长的蜂鸣声
            for _ in range(3):  # 默认响三声
                if self.stop_sound:
                    break
                winsound.Beep(AlarmConfig.ALARM_FREQUENCY, AlarmConfig.ALARM_DURATION)
                time.sleep(0.1)
        except ImportError:
            # 非Windows系统（如Linux/Mac），尝试使用系统控制台铃声
            print('\a')
        except Exception as e:
            logger.error(f"播放报警声音失败: {e}")
    
    def stop_alarm(self):
        """停止报警"""
        self.stop_sound = True
        self.is_alarming = False
        self.current_level = AlarmConfig.LEVEL_NORMAL
        self.violation_count = 0
    
    def register_callback(self, callback):
        """
        注册报警回调函数
        
        Args:
            callback: 回调函数，接收event字典作为参数
        """
        self.alarm_callbacks.append(callback)
    
    def get_status_color(self):
        """
        获取当前状态对应的颜色
        
        Returns:
            tuple: BGR颜色元组
        """
        if self.current_level == AlarmConfig.LEVEL_NORMAL:
            return AlarmConfig.NORMAL_COLOR
        elif self.current_level == AlarmConfig.LEVEL_WARNING:
            return AlarmConfig.WARNING_COLOR
        else:
            return AlarmConfig.DANGER_COLOR
    
    def get_status_text(self):
        """
        获取当前状态文本
        
        Returns:
            str: 状态文本
        """
        if self.current_level == AlarmConfig.LEVEL_NORMAL:
            return "正常监控中"
        elif self.current_level == AlarmConfig.LEVEL_WARNING:
            return "⚠️ 警告: 检测到可疑操作"
        else:
            return "🚨 危险: 检测到违规取放!"
    
    def get_recent_events(self, count=10):
        """
        获取最近的事件记录
        
        Args:
            count: 获取数量
            
        Returns:
            list: 事件列表
        """
        return self.events[-count:]
    
    def save_screenshot(self, frame, prefix="violation"):
        """
        保存违规截图
        
        Args:
            frame: 图像帧
            prefix: 文件名前缀
            
        Returns:
            str: 保存的文件路径
        """
        import cv2
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        
        cv2.imwrite(filepath, frame)
        logger.info(f"截图已保存: {filepath}")
        
        return filepath
    
    def reset(self):
        """重置报警系统状态"""
        self.current_level = AlarmConfig.LEVEL_NORMAL
        self.violation_count = 0
        self.is_alarming = False
        self.stop_sound = True


class EventLogger:
    """：负责将违规事件持久化存储到本地CSV文件中。
    """
    
    def __init__(self, log_file=None):
        """初始化记录器，创建日志文件并写入表头"""
        if log_file is None:
            # 默认生成以当前时间命名的CSV文件
            log_file = os.path.join(
                LOG_DIR, 
                f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        self.log_file = log_file
        
        # 初始化CSV文件结构
        self._write_header()
    
    def _write_header(self):
        """写入CSV文件头，定义数据列"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("时间,级别,置信度,消息,截图路径\n")
    
    def log_event(self, event, screenshot_path=None):
        """
        将单条事件记录到CSV文件中。
        
        Args:
            event (dict): 包含时间、级别等信息的字典
            screenshot_path (str): 对应的违规截图路径
        """
        # 路径处理：将绝对路径转换为相对路径，方便在不同电脑上查看
        if screenshot_path and os.path.isabs(screenshot_path):
            from config import BASE_DIR
            try:
                screenshot_path = os.path.relpath(screenshot_path, BASE_DIR)
            except ValueError:
                pass # 跨驱动器时可能失败，保持原样
            
        with open(self.log_file, 'a', encoding='utf-8') as f:
            time_str = event['time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            # 使用CSV标准格式写入，消息内容加引号防止逗号冲突
            time_str = event['time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(f"{time_str},{event['level']},{event['confidence']:.4f},")
            f.write(f"\"{event['message']}\",{screenshot_path or ''}\n")
