# -*- coding: utf-8 -*-
"""
生产线违规取放检测系统 - GUI界面
Production Line Violation Detection System - GUI Interface
"""

import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import queue

from main import ViolationDetectionSystem
from config import NORMAL_VIDEO_DIR, ABNORMAL_VIDEO_DIR, VIDEO_DIR, AlarmConfig

class DetectionGUI:
    """
    系统图形用户界面 (GUI)：
    基于 Tkinter 构建，采用多线程架构确保视频处理与界面响应互不干扰。
    实现了视频预览、实时状态监控、报警日志记录及示例视频快速切换功能。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("生产线违规取放检测系统 v1.0")
        self.root.geometry("1200x800")
        
        # 初始化核心检测系统
        self.system = ViolationDetectionSystem()
        self.is_running = False
        self.current_video_path = None
        
        # 线程间通信队列：用于将检测线程处理后的帧安全地传递给 UI 线程显示
        # maxsize=2 限制缓冲区大小，确保显示的画面具有最低延迟
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None  # 后台检测线程引用
        
        self._setup_ui()
        
        # 观察者模式：将 GUI 的日志更新函数注册到报警系统的回调列表中
        self.system.alarm_system.register_callback(self._on_alarm_triggered)
        
        # 启动 UI 刷新定时器：每 15ms 检查一次队列是否有新画面
        self.root.after(10, self._poll_frame_queue)

    def _poll_frame_queue(self):
        """
        UI 线程轮询函数：负责从队列中提取最新帧并渲染。
        这是解决“多线程操作 UI 崩溃”问题的标准做法。
        """
        try:
            frame = None
            # 尽可能获取队列中最新的帧，舍弃旧帧
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame is not None:
                self._display_frame(frame)
        except queue.Empty:
            pass
        
        # 递归调用，维持 UI 刷新循环
        self.root.after(15, self._poll_frame_queue)

    def _display_frame(self, frame):
        """
        图像格式转换与缩放：将 OpenCV 的 BGR 格式转换为 Tkinter 可显示的 PhotoImage。
        """
        # 颜色空间转换：OpenCV (BGR) -> PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # 自适应缩放：保持比例缩放到显示区域大小
        max_h, max_w = 600, 800
        scale = min(max_h/h, max_w/w)
        frame_resized = cv2.resize(frame_rgb, (int(w*scale), int(h*scale)))
        
        # 转换为 Tkinter 兼容对象
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk # 必须保持引用，否则会被垃圾回收导致白屏
        self.video_label.configure(image=imgtk)

    def _setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧视频显示区
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(left_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧控制面板
        right_panel = ttk.Frame(main_frame, width=300, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 视频源选择
        source_group = ttk.LabelFrame(right_panel, text="视频源控制", padding="10")
        source_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(source_group, text="选择视频文件", command=self._select_video).pack(fill=tk.X, pady=2)
        
        # 示例视频快捷选择
        example_group = ttk.LabelFrame(right_panel, text="示例视频", padding="10")
        example_group.pack(fill=tk.X, pady=5)
        
        self.example_var = tk.StringVar()
        self.example_combo = ttk.Combobox(example_group, textvariable=self.example_var)
        self._refresh_examples()
        self.example_combo.pack(fill=tk.X, pady=2)
        ttk.Button(example_group, text="播放示例", command=self._play_example).pack(fill=tk.X, pady=2)
        
        # 系统控制
        control_group = ttk.LabelFrame(right_panel, text="系统控制", padding="10")
        control_group.pack(fill=tk.X, pady=5)
        
        self.btn_pause = ttk.Button(control_group, text="暂停", command=self._toggle_pause)
        self.btn_pause.pack(fill=tk.X, pady=2)
        ttk.Button(control_group, text="停止", command=self._stop_detection).pack(fill=tk.X, pady=2)
        ttk.Button(control_group, text="重置报警", command=self._reset_alarm).pack(fill=tk.X, pady=2)
        
        # 状态显示
        status_group = ttk.LabelFrame(right_panel, text="实时状态", padding="10")
        status_group.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="等待启动...")
        ttk.Label(status_group, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(pady=5)
        
        self.stats_var = tk.StringVar(value="报警次数: 0\n违规帧数: 0")
        ttk.Label(status_group, textvariable=self.stats_var, justify=tk.LEFT).pack(pady=5)
        
        # 报警日志
        log_group = ttk.LabelFrame(right_panel, text="报警日志", padding="10")
        log_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_list = tk.Listbox(log_group, height=10)
        self.log_list.pack(fill=tk.BOTH, expand=True)
        
    def _refresh_examples(self):
        examples = []
        if os.path.exists(ABNORMAL_VIDEO_DIR):
            examples.extend([os.path.join("异常", f) for f in os.listdir(ABNORMAL_VIDEO_DIR) if f.endswith('.mp4')])
        if os.path.exists(NORMAL_VIDEO_DIR):
            examples.extend([os.path.join("正常", f) for f in os.listdir(NORMAL_VIDEO_DIR) if f.endswith('.mp4')])
        self.example_combo['values'] = examples

    def _on_alarm_triggered(self, event):
        """当报警触发时的GUI反馈"""
        time_str = event['time'].strftime('%H:%M:%S')
        msg = f"[{time_str}] {event['message']}"
        self.log_list.insert(0, msg)
        if self.log_list.size() > 50:
            self.log_list.delete(50, tk.END)
        
        # 弹出提示框 (可选，如果不想打断操作可以只在界面显示)
        # messagebox.showwarning("违规警报", f"检测到违规操作！\n时间: {time_str}")

    def _select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if path:
            self._stop_detection(wait=True)
            self.current_video_path = path
            self._start_thread()

    def _play_example(self):
        rel_path = self.example_var.get()
        if rel_path:
            path = os.path.join(VIDEO_DIR, rel_path)
            self._stop_detection(wait=True)
            self.current_video_path = path
            self._start_thread()

    def _start_thread(self):
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # 重置系统状态
        self.system.alarm_system.reset()
        self.system.hand_detector.detection_history = []
        self.system.hand_detector.prev_frame = None
        self.system.hand_detector.motion_position_history = []
        
        self.is_running = True
        self.system.is_paused = False
        self.btn_pause.config(text="暂停")
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def _detection_loop(self):
        """
        后台检测线程主循环：
        负责视频读取、算法推理、结果渲染及状态同步。
        """
        from video_processor import VideoProcessor
        self.system.video_processor = VideoProcessor(self.current_video_path)
        try:
            self.system.video_processor.open()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开视频源: {e}")
            return

        # 动态获取视频尺寸并计算 ROI 区域
        width = self.system.video_processor.width
        height = self.system.video_processor.height
        self.system.roi = self.system._get_roi(width, height)
        
        # 初始化统计数据
        self.system.stats = {'total_frames': 0, 'violation_frames': 0, 'total_alarms': 0}
        
        while self.is_running:
            if not self.system.is_paused:
                # 1. 读取原始帧
                ret, frame = self.system.video_processor.read()
                if not ret or frame is None:
                    break
                
                self.system.current_frame = frame.copy()
                self.system.stats['total_frames'] += 1
                
                # 2. 执行多模态检测算法
                res = self.system.hand_detector.detect(frame, self.system.roi)
                
                # 3. 更新报警状态机
                level = self.system.alarm_system.update(res['stable_detection'], res['detection_confidence'])
                
                if res['stable_detection']:
                    self.system.stats['violation_frames'] += 1
                
                # 4. 视觉渲染：叠加 ROI、检测框及报警特效
                frame = self.system.hand_detector.draw_results(frame, res)
                frame = self.system.frame_renderer.draw_roi(frame, self.system.roi)
                frame = self.system.frame_renderer.draw_alarm_overlay(frame, level)
                
                # 5. 跨线程同步：将处理后的画面和状态发送给 UI 线程
                self._update_frame_on_gui(frame)
                self._update_status_on_gui()
            else:
                # 暂停状态下释放 CPU 资源
                time.sleep(0.1)
                
        self.system.video_processor.release()
        self.status_var.set("已停止")

    def _update_frame_on_gui(self, frame):
        """将帧放入队列，由检测线程调用"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            except queue.Empty:
                self.frame_queue.put(frame)

    def _update_status_on_gui(self):
        status_text = self.system.alarm_system.get_status_text()
        self.status_var.set(status_text)
        
        stats_text = f"报警次数: {self.system.stats['total_alarms']}\n"
        stats_text += f"违规帧数: {self.system.stats['violation_frames']}\n"
        stats_text += f"总帧数: {self.system.stats['total_frames']}"
        self.stats_var.set(stats_text)

    def _toggle_pause(self):
        self.system.is_paused = not self.system.is_paused
        self.btn_pause.config(text="继续" if self.system.is_paused else "暂停")

    def _stop_detection(self, wait=False):
        self.is_running = False
        self.system.alarm_system.reset()
        
        # 等待线程结束以避免卡壳
        if wait and self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _reset_alarm(self):
        self.system.alarm_system.reset()
        self.log_list.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] 系统重置")

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()
