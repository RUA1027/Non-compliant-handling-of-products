
import cv2
import numpy as np
from detector import ViolationDetector
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_frames():
    detector = ViolationDetector(history=500, var_threshold=25, min_area=1000)
    
    base_dir = Path(__file__).resolve().parent
    videos_dir = base_dir / "视频素材"
    
    # Check if directories exist
    if not videos_dir.exists():
        print(f"Error: Video directory not found at {videos_dir}")
        return

    # Dynamically find first normal and abnormal video for analysis
    normal_videos = list((videos_dir / "正常").glob("*.mp4"))
    abnormal_videos = list((videos_dir / "异常").glob("*.mp4"))
    
    videos = []
    if normal_videos: videos.append(str(normal_videos[0]))
    if abnormal_videos: videos.append(str(abnormal_videos[0]))
    
    if not videos:
        print("No videos found to analyze.")
        return
    
    results = {}
    
    for v_path in videos:
        print(f"Analyzing {v_path}...")
        cap = cv2.VideoCapture(v_path)
        frame_areas = []
        
        # Reset detector for each video manually since we are using internal method
        detector.heatmap = None
        detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=detector.history, 
            varThreshold=detector.var_threshold, 
            detectShadows=True
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            _, _, contours = detector.detect_motion(frame, roi=None)
            
            total_area = 0
            for cnt in contours:
                total_area += cv2.contourArea(cnt)
            frame_areas.append(total_area)
            
        cap.release()
        results[v_path] = frame_areas
        
        print(f"Stats for {v_path}:")
        print(f"  Max Area: {max(frame_areas)}")
        print(f"  Mean Area: {np.mean(frame_areas)}")
        print(f"  90th Percentile: {np.percentile(frame_areas, 90)}")
        print("-" * 20)

if __name__ == "__main__":
    analyze_frames()
