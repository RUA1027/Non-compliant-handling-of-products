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
    报警系统类
    负责管理报警状态、触发报警、记录事件
    """
    
    def __init__(self):
        """初始化报警系统"""
        self.current_level = AlarmConfig.LEVEL_NORMAL
        self.violation_count = 0
        self.last_alarm_time = 0
        self.is_alarming = False
        
        # 事件记录
        self.events = []
        self.max_events = 100
        
        # 报警回调函数
        self.alarm_callbacks = []
        
        # 声音报警线程
        self.sound_thread = None
        self.stop_sound = False
        
        logger.info("报警系统初始化完成")
    
    def update(self, hand_detected, confidence=0.0):
        """
        更新报警状态
        
        Args:
            hand_detected: 是否检测到手部
            confidence: 检测置信度
            
        Returns:
            int: 当前报警级别
        """
        current_time = time.time()
        
        if hand_detected:
            self.violation_count += 1
            
            # 根据连续检测帧数判断报警级别
            if self.violation_count >= AlarmConfig.ALARM_FRAME_THRESHOLD:
                # 检查冷却时间
                if current_time - self.last_alarm_time >= AlarmConfig.ALARM_COOLDOWN_SECONDS:
                    self.current_level = AlarmConfig.LEVEL_DANGER
                    self._trigger_alarm(confidence)
                    self.last_alarm_time = current_time
            elif self.violation_count >= AlarmConfig.ALARM_FRAME_THRESHOLD // 2:
                self.current_level = AlarmConfig.LEVEL_WARNING
        else:
            # 立即重置违规计数和报警状态，实现“撤销及时”
            self.violation_count = 0
            self.current_level = AlarmConfig.LEVEL_NORMAL
            self.is_alarming = False
        
        return self.current_level
    
    def _trigger_alarm(self, confidence):
        """
        触发报警
        
        Args:
            confidence: 检测置信度
        """
        self.is_alarming = True
        
        # 记录事件
        event = {
            'time': datetime.now(),
            'level': 'DANGER',
            'confidence': confidence,
            'message': '检测到违规取放操作!'
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        logger.warning(f"⚠️ 违规警报: 检测到手部操作! 置信度: {confidence:.2f}")
        
        # 执行回调
        for callback in self.alarm_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"报警回调执行失败: {e}")
        
        # 触发声音报警
        if AlarmConfig.SOUND_ENABLED:
            self._play_alarm_sound()
    
    def _play_alarm_sound(self):
        """播放报警声音"""
        if self.sound_thread is not None and self.sound_thread.is_alive():
            return
        
        self.stop_sound = False
        self.sound_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self.sound_thread.start()
    
    def _sound_worker(self):
        """声音播放工作线程"""
        try:
            import winsound
            for _ in range(3):  # 响三声
                if self.stop_sound:
                    break
                winsound.Beep(AlarmConfig.ALARM_FREQUENCY, AlarmConfig.ALARM_DURATION)
                time.sleep(0.1)
        except ImportError:
            # 非Windows系统，使用系统铃声
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
    """
    事件日志记录器
    """
    
    def __init__(self, log_file=None):
        """初始化日志记录器"""
        if log_file is None:
            log_file = os.path.join(
                LOG_DIR, 
                f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        self.log_file = log_file
        
        # 创建CSV文件头
        self._write_header()
    
    def _write_header(self):
        """写入CSV文件头"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("时间,级别,置信度,消息,截图路径\n")
    
    def log_event(self, event, screenshot_path=None):
        """
        记录事件
        
        Args:
            event: 事件字典
            screenshot_path: 截图路径
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            time_str = event['time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(f"{time_str},{event['level']},{event['confidence']:.4f},")
            f.write(f"\"{event['message']}\",{screenshot_path or ''}\n")
