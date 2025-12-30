# -*- coding: utf-8 -*-
"""
主程序入口模块 (Main Entry Module)
--------------------------------------------------
【作用与功能】
本模块是系统的“编排者 (Orchestrator)”，负责组装各子模块并运行主业务流程。
主要功能包括：命令行参数解析、批量测试执行、单视频/摄像头模式调度。

【使用的工具与技术】
1. Argparse：解析命令行参数 (-v, -c, --test)。
2. 模块化组装：实例化 HandDetector, AlarmSystem, VideoProcessor 并串联逻辑。
3. 异常处理：管理视频流的打开与资源释放。

【实现方式】
- ViolationDetectionSystem 类：核心业务类，封装了 process_video 主循环。
  在循环中依次调用：读取帧 -> 检测 -> 报警更新 -> 渲染 -> 显示/保存。
- main 函数：根据用户输入选择运行模式（GUI/CLI/Test）。
--------------------------------------------------
"""

import os
import sys
import cv2
import time
import argparse
from datetime import datetime

from config import (
    DetectionConfig, AlarmConfig, DisplayConfig, VideoConfig,
    NORMAL_VIDEO_DIR, ABNORMAL_VIDEO_DIR, LOG_DIR, SCREENSHOT_DIR
)
from hand_detector import HandDetector
from alarm_system import AlarmSystem, EventLogger
from video_processor import VideoProcessor, FrameRenderer


class ViolationDetectionSystem:
    """
    违规取放检测系统主类
    """
    
    def __init__(self):
        """初始化检测系统"""
        self.hand_detector = HandDetector()
        self.alarm_system = AlarmSystem()
        self.video_processor = None
        self.frame_renderer = FrameRenderer()
        self.event_logger = None
        self.current_frame = None
        
        # 状态变量
        self.is_running = False
        self.is_paused = False
        self.roi = None
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'violation_frames': 0,
            'total_alarms': 0
        }
        
        # 注册报警回调
        self.alarm_system.register_callback(self._on_alarm)
        
        print("=" * 60)
        print("生产线违规取放检测系统 v1.0")
        print("Production Line Violation Detection System")
        print("=" * 60)
    
    def _on_alarm(self, event):
        """报警触发回调"""
        self.stats['total_alarms'] += 1
        
        # 保存截图
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            screenshot_path = self.alarm_system.save_screenshot(self.current_frame)
            if self.event_logger:
                self.event_logger.log_event(event, screenshot_path)
    
    def _get_roi(self, frame_width, frame_height):
        """计算ROI区域"""
        if not DetectionConfig.ROI_ENABLED:
            return None
        
        x = int(DetectionConfig.ROI_X * frame_width)
        y = int(DetectionConfig.ROI_Y * frame_height)
        w = int(DetectionConfig.ROI_WIDTH * frame_width)
        h = int(DetectionConfig.ROI_HEIGHT * frame_height)
        
        return (x, y, w, h)
    
    def process_video(self, video_source, show_window=True, save_result=False):
        """
        处理视频
        
        Args:
            video_source: 视频源（文件路径或摄像头索引）
            show_window: 是否显示窗口
            save_result: 是否保存处理结果
        """
        # 初始化视频处理器
        self.video_processor = VideoProcessor(video_source)
        
        try:
            self.video_processor.open()
        except ValueError as e:
            print(f"错误: {e}")
            return
        
        # 初始化事件日志
        self.event_logger = EventLogger()
        
        # 计算ROI
        self.roi = self._get_roi(self.video_processor.width, self.video_processor.height)
        
        # 开始录制（如果启用）
        if save_result:
            output_path = os.path.join(
                LOG_DIR, 
                f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            self.video_processor.start_recording(output_path)
            print(f"录制输出: {output_path}")
        
        self.is_running = True
        print(f"\n开始处理视频: {video_source}")
        print("按 'q' 退出, 'p' 暂停/继续, 's' 保存截图, 'r' 重置报警")
        print("-" * 60)
        
        while self.is_running:
            if not self.is_paused:
                ret, frame = self.video_processor.read()
                
                if not ret:
                    # 视频结束
                    print("\n视频播放完毕")
                    break
                
                self.current_frame = frame.copy()
                self.stats['total_frames'] += 1
                
                # 手部检测
                detection_result = self.hand_detector.detect(frame, self.roi)
                
                # 更新报警状态
                alarm_level = self.alarm_system.update(
                    detection_result['stable_detection'],
                    detection_result['detection_confidence']
                )
                
                if detection_result['stable_detection']:
                    self.stats['violation_frames'] += 1
                
                # 绘制检测结果
                frame = self.hand_detector.draw_results(
                    frame, 
                    detection_result,
                    show_landmarks=DisplayConfig.SHOW_HAND_LANDMARKS,
                    show_boxes=DisplayConfig.SHOW_DETECTION_BOX
                )
                
                # 绘制ROI
                frame = self.frame_renderer.draw_roi(frame, self.roi)
                
                # 绘制报警覆盖层
                frame = self.frame_renderer.draw_alarm_overlay(frame, alarm_level)
                
                # 绘制状态栏
                frame = self.frame_renderer.draw_status_bar(
                    frame,
                    self.alarm_system.get_status_text(),
                    self.alarm_system.get_status_color(),
                    self.video_processor.fps
                )
                
                # 绘制信息面板
                info = {
                    'Confidence': f"{detection_result['detection_confidence']:.2f}",
                    'Violations': self.stats['violation_frames'],
                    'Alarms': self.stats['total_alarms'],
                    'Progress': f"{self.video_processor.get_progress()*100:.1f}%"
                }
                frame = self.frame_renderer.draw_info_panel(frame, info)
                
                # 录制
                if save_result:
                    self.video_processor.write_frame(frame)
                
                # 显示
                if show_window:
                    cv2.imshow('Violation Detection', frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
                print(f"{'暂停' if self.is_paused else '继续'}")
            elif key == ord('s'):
                self.alarm_system.save_screenshot(self.current_frame, "manual")
                print("截图已保存")
            elif key == ord('r'):
                self.alarm_system.reset()
                print("报警系统已重置")
        
        # 清理
        self.is_running = False
        self.video_processor.release()
        self.hand_detector.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("处理统计")
        print("-" * 60)
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"违规帧数: {self.stats['violation_frames']}")
        if self.stats['total_frames'] > 0:
            violation_rate = self.stats['violation_frames'] / self.stats['total_frames'] * 100
            print(f"违规率: {violation_rate:.2f}%")
        print(f"触发报警次数: {self.stats['total_alarms']}")
        print("=" * 60)
    
    def process_realtime(self, camera_index=0):
        """
        实时处理摄像头视频
        
        Args:
            camera_index: 摄像头索引
        """
        print(f"\n正在打开摄像头 {camera_index}...")
        self.process_video(camera_index, show_window=True, save_result=False)


def test_all_videos():
    """测试所有视频文件"""
    system = ViolationDetectionSystem()
    
    # 测试异常视频
    print("\n" + "=" * 60)
    print("测试异常视频")
    print("=" * 60)
    
    abnormal_videos = [f for f in os.listdir(ABNORMAL_VIDEO_DIR) if f.endswith('.mp4')]
    for video_file in sorted(abnormal_videos):
        video_path = os.path.join(ABNORMAL_VIDEO_DIR, video_file)
        print(f"\n>>> 正在处理: {video_file}")
        
        # 重新初始化系统
        system = ViolationDetectionSystem()
        system.process_video(video_path, show_window=True, save_result=False)
    
    # 测试正常视频
    print("\n" + "=" * 60)
    print("测试正常视频")
    print("=" * 60)
    
    normal_videos = [f for f in os.listdir(NORMAL_VIDEO_DIR) if f.endswith('.mp4')]
    for video_file in sorted(normal_videos):
        video_path = os.path.join(NORMAL_VIDEO_DIR, video_file)
        print(f"\n>>> 正在处理: {video_file}")
        
        # 重新初始化系统
        system = ViolationDetectionSystem()
        system.process_video(video_path, show_window=True, save_result=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生产线违规取放检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 使用默认摄像头
  python main.py -v 视频素材/异常/异常1.mp4  # 处理指定视频
  python main.py -c 1                     # 使用摄像头1
  python main.py --test                   # 测试所有视频
        """
    )
    
    parser.add_argument('-v', '--video', type=str, default=None,
                       help='视频文件路径')
    parser.add_argument('-c', '--camera', type=int, default=None,
                       help='摄像头索引')
    parser.add_argument('--test', action='store_true',
                       help='测试所有示例视频')
    parser.add_argument('--save', action='store_true',
                       help='保存处理结果')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示窗口（后台处理）')
    
    args = parser.parse_args()
    
    if args.test:
        test_all_videos()
    elif args.video:
        system = ViolationDetectionSystem()
        system.process_video(
            args.video, 
            show_window=not args.no_display,
            save_result=args.save
        )
    elif args.camera is not None:
        system = ViolationDetectionSystem()
        system.process_realtime(args.camera)
    else:
        # 默认：显示菜单
        print("\n请选择运行模式:")
        print("1. 摄像头实时检测")
        print("2. 视频文件检测")
        print("3. 测试所有示例视频")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            system = ViolationDetectionSystem()
            system.process_realtime(0)
        elif choice == '2':
            video_path = input("请输入视频文件路径: ").strip()
            if os.path.exists(video_path):
                system = ViolationDetectionSystem()
                system.process_video(video_path)
            else:
                print(f"文件不存在: {video_path}")
        elif choice == '3':
            test_all_videos()
        elif choice == '4':
            print("再见!")
        else:
            print("无效选项")


if __name__ == "__main__":
    main()
