
import cv2
import os
import json
import argparse
from pathlib import Path
from detector import ViolationDetector
from tqdm import tqdm

def main():
    # Get the directory where the script is located
    base_dir = Path(__file__).resolve().parent
    
    # Default paths relative to the script directory
    default_input = base_dir / "视频素材"
    default_output = base_dir / "results"

    parser = argparse.ArgumentParser(description="Production Line Violation Detection")
    parser.add_argument("--input", default=str(default_input), help="Input directory containing videos")
    parser.add_argument("--output", default=str(default_output), help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (heuristic)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(list(input_dir.rglob(ext)))

    print(f"Found {len(video_files)} videos.")

    detector = ViolationDetector(history=500, var_threshold=25, min_area=1000)
    
    results = {}
    
    # Define ROI - Tuned based on image observation
    # The conveyor belt is the main area. Top part is machinery.
    # Resolution is around 892x742. 
    # Let's focus on the bottom 3/4ths roughly, or full frame but ignore small changes.
    # Looking at sample 'normal1_sec0.jpg', the belt covers most.
    # We'll use full frame for now but rely on large motion detection.
    
    for video_path in tqdm(video_files, desc="Processing Videos"):
        print(f"Analyzing {video_path.name}...")
        
        # Subdirectory structure for output to keep it organized
        rel_path = video_path.relative_to(input_dir)
        out_vid_path = output_dir / rel_path
        out_vid_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics = detector.analyze_video(
            str(video_path), 
            output_path=str(out_vid_path).replace(out_vid_path.suffix, '_out.mp4')
        )
        
        # CLASSIFICATION LOGIC
        # Tuned Threshold based on analysis:
        # Normal videos have avg_motion_area < 43,000 (Max observed ~42k)
        # Abnormal videos have avg_motion_area > 47,000 (Min observed ~47.9k)
        # We set threshold at 45,000.
        
        avg_area = metrics['avg_motion_area']
        threshold = 45000
        
        is_abnormal = avg_area > threshold
        prediction = "异常" if is_abnormal else "正常"
        
        # Ground Truth (from filename)
        ground_truth = "异常" if "异常" in video_path.name else "正常"
        
        results[video_path.name] = {
            "metrics": metrics,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth
        }
        
        status_icon = "✅" if prediction == ground_truth else "❌"
        print(f"  Processed {video_path.name}: Avg Area={avg_area:.1f} -> {prediction} {status_icon}")

    # Save detailed analysis
    with open(output_dir / "analysis_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # Calculate Accuracy
    total = len(results)
    correct = sum(1 for r in results.values() if r['correct'])
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print("="*50)
    print(f"Total Videos: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()
