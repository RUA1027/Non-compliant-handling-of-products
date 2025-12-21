
import json
import os
from pathlib import Path

def verify_results():
    base_dir = Path(__file__).resolve().parent
    metrics_path = base_dir / "results" / "analysis_metrics.json"
    if not os.path.exists(metrics_path):
        print("Metrics file not found. Please run main.py first.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    threshold = 45000
    correct = 0
    total = 0

    print(f"{'Video':<15} | {'Avg Area':<10} | {'Pred':<5} | {'True':<5} | {'Result'}")
    print("-" * 60)

    for filename, metrics in data.items():
        avg_area = metrics['avg_motion_area']
        is_abnormal = avg_area > threshold
        prediction = "异常" if is_abnormal else "正常"
        ground_truth = "异常" if "异常" in filename else "正常"
        
        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        total += 1
        
        mark = "OK" if is_correct else "FAIL"
        print(f"{filename:<15} | {avg_area:<10.1f} | {prediction:<5} | {ground_truth:<5} | {mark}")

    print("-" * 60)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    verify_results()
