
import cv2
import numpy as np
from pathlib import Path

class ViolationDetector:
    def __init__(self, history=500, var_threshold=25, min_area=500):
        """
        Initialize the violation detector.
        
        Args:
            history (int): Length of the history for background subtraction.
            var_threshold (int): Threshold on the squared Mahalanobis distance between the pixel and the model.
            min_area (int): Minimum contour area to be considered as valid motion.
        """
        self.history = history
        self.var_threshold = var_threshold
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=True
        )
        self.min_area = min_area
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.heatmap = None

    def detect_motion(self, frame, roi=None):
        """
        Detect motion in the frame.
        
        Args:
            frame (np.ndarray): Input video frame.
            roi (tuple): Optional ROI (x, y, w, h).
            
        Returns:
            processed_frame (np.ndarray): Frame with annotations.
            mask (np.ndarray): Foreground mask.
            contours (list): List of detected motion contours.
        """
        if roi:
            x, y, w, h = roi
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_roi)
        
        # Remove shadows (shadows are gray in MOG2, we want only white foreground)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                filtered_contours.append(cnt)
                
        return frame_roi, fg_mask, filtered_contours

    def analyze_video(self, video_path, output_path=None, roi=None):
        """
        Analyze a video for violation detection.
        
        Args:
            video_path (str): Path to input video.
            output_path (str): Path to save annotated video.
            roi (tuple): Region of interest (x, y, w, h).
            
        Returns:
            result (dict): Analysis results and metrics.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define ROI if not provided (default to full frame or specific area)
        # Based on sample images, the action happens mostly in the center-right
        if roi is None:
            # Default to full frame for now, user can refine
            roi = (0, 0, width, height)

        # RESET STATE for new video
        self.heatmap = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history, 
            varThreshold=self.var_threshold, 
            detectShadows=True
        )
            
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
        frame_idx = 0
        
        # Statistics
        motion_frames = 0
        max_motion_area = 0
        total_motion_area = 0
        motion_centroids = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_roi, mask, contours = self.detect_motion(frame, roi)
            
            # Accumulate heatmap (binary mask -> float accumulation)
            if mask is not None:
                # Normalize mask to 0-1 and add to heatmap
                mask_norm = mask.astype(np.float32) / 255.0
                if self.heatmap is None:
                    self.heatmap = np.zeros_like(mask_norm)
                self.heatmap += mask_norm
            
            # Draw ROI on original frame (for visualization)
            if roi != (0, 0, width, height):
                rx, ry, rw, rh = roi
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
            
            # Calculate total area for this frame to decide Warning Status
            frame_total_area = sum(cv2.contourArea(c) for c in contours)
            alarm_threshold = 150000 # Tuned Threshold for instantaneous alarm
            
            is_violation = frame_total_area > alarm_threshold
            color = (0, 0, 255) if is_violation else (0, 255, 0) # Red if violation, Green if normal
            status_text = "WARNING: VIOLATION DETECTED" if is_violation else "STATUS: NORMAL"
            
            current_frame_max_area = 0
            if contours:
                motion_frames += 1
                
                # Combine all motion contours to get total area/centroid
                all_points = np.vstack(contours)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    current_frame_max_area = max(current_frame_max_area, area)
                    total_motion_area += area
                    
                    # Draw contour
                    # Adjust coordinates if ROI was used
                    draw_cnt = cnt
                    if roi:
                        draw_cnt = cnt + np.array([[roi[0], roi[1]]])
                        
                    cv2.drawContours(frame, [draw_cnt], -1, color, 2)
                    
                    x, y, w, h = cv2.boundingRect(draw_cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Calculate centroid of the largest motion
                if len(all_points) > 0:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if roi:
                            cX += roi[0]
                            cY += roi[1]
                        motion_centroids.append((cX, cY))
                        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            max_motion_area = max(max_motion_area, current_frame_max_area)
            
            # Draw Status Text
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, color, 3)
            
            if writer:
                writer.write(frame)
                
            frame_idx += 1
            
        cap.release()
        if writer:
            writer.release()
            
        # Post-process Heatmap
        heatmap_metrics = {}
        if self.heatmap is not None:
            # Normalize heatmap for visualization 0-255
            heatmap_vis = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            
            # Save heatmap
            if output_path:
                heatmap_path = str(output_path).replace('.mp4', '_heatmap.jpg')
                cv2.imwrite(heatmap_path, heatmap_color)
            
            # Grid Analysis (3x3)
            h, w = self.heatmap.shape
            dy, dx = h // 3, w // 3
            grid_energy = []
            for r in range(3):
                for c in range(3):
                    cell = self.heatmap[r*dy:(r+1)*dy, c*dx:(c+1)*dx]
                    energy = np.sum(cell)
                    grid_energy.append(energy)
            
            # Normalize grid energy by total energy
            total_energy = np.sum(self.heatmap) if np.sum(self.heatmap) > 0 else 1
            grid_prob = [float(e / total_energy) for e in grid_energy]
            heatmap_metrics = {
                "grid_energy_distribution": grid_prob, # List of 9 values (Top-Left to Bottom-Right)
                "total_motion_energy": float(total_energy)
            }
            
        # Calculate some higher-level features
        motion_ratio = motion_frames / total_frames if total_frames > 0 else 0
        
        return {
            "total_frames": total_frames,
            "motion_frames": motion_frames,
            "motion_ratio": motion_ratio,
            "max_motion_area": max_motion_area,
            "avg_motion_area": total_motion_area / motion_frames if motion_frames > 0 else 0,
            "centroids_count": len(motion_centroids),
            "heatmap_metrics": heatmap_metrics
        }

if __name__ == "__main__":
    # Simple test
    detector = ViolationDetector()
    print("Detector initialized. Run via main.py")
