# -*- coding: utf-8 -*-
"""
生产线违规取放检测系统 - GUI界面
Production Line Violation Detection System - GUI Interface
"""

import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime

from main import ViolationDetectionSystem
from config import NORMAL_VIDEO_DIR, ABNORMAL_VIDEO_DIR, VIDEO_DIR, AlarmConfig

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("生产线违规取放检测系统 v1.0")
        self.root.geometry("1200x800")
        
        self.system = ViolationDetectionSystem()
        self.is_running = False
        self.current_video_path = None
        
        self._setup_ui()
        
        # 注册报警回调以更新GUI
        self.system.alarm_system.register_callback(self._on_alarm_triggered)

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
        
        ttk.Button(source_group, text="打开摄像头", command=self._start_camera).pack(fill=tk.X, pady=2)
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

    def _start_camera(self):
        self._stop_detection()
        self.current_video_path = 0
        self._start_thread()

    def _select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if path:
            self._stop_detection()
            self.current_video_path = path
            self._start_thread()

    def _play_example(self):
        rel_path = self.example_var.get()
        if rel_path:
            path = os.path.join(VIDEO_DIR, rel_path)
            self._stop_detection()
            self.current_video_path = path
            self._start_thread()

    def _start_thread(self):
        self.is_running = True
        self.system.is_paused = False
        self.btn_pause.config(text="暂停")
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def _detection_loop(self):
        from video_processor import VideoProcessor
        self.system.video_processor = VideoProcessor(self.current_video_path)
        try:
            self.system.video_processor.open()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开视频源: {e}")
            return

        # 获取视频尺寸用于ROI计算
        width = self.system.video_processor.width
        height = self.system.video_processor.height
        self.system.roi = self.system._get_roi(width, height)
        
        # 重置统计
        self.system.stats = {'total_frames': 0, 'violation_frames': 0, 'total_alarms': 0}
        
        while self.is_running:
            if not self.system.is_paused:
                ret, frame = self.system.video_processor.read()
                if not ret or frame is None:
                    break
                
                self.system.current_frame = frame.copy()
                self.system.stats['total_frames'] += 1
                
                # 检测
                res = self.system.hand_detector.detect(frame, self.system.roi)
                level = self.system.alarm_system.update(res['stable_detection'], res['detection_confidence'])
                
                if res['stable_detection']:
                    self.system.stats['violation_frames'] += 1
                
                # 渲染
                frame = self.system.hand_detector.draw_results(frame, res)
                frame = self.system.frame_renderer.draw_roi(frame, self.system.roi)
                frame = self.system.frame_renderer.draw_alarm_overlay(frame, level)
                
                # 更新GUI显示
                self._update_frame_on_gui(frame)
                self._update_status_on_gui()
            else:
                self.root.update()
                cv2.waitKey(30)
                
        self.system.video_processor.release()
        self.status_var.set("已停止")

    def _update_frame_on_gui(self, frame):
        # 缩放以适应显示区域
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        max_h, max_w = 600, 800
        scale = min(max_h/h, max_w/w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

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

    def _stop_detection(self):
        self.is_running = False
        self.system.alarm_system.reset()

    def _reset_alarm(self):
        self.system.alarm_system.reset()
        self.log_list.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] 系统重置")

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()
