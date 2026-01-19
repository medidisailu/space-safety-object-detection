from ultralytics import YOLO
import numpy as np
import pandas as pd

# ✅ Load trained model
model = YOLO("D:/ml2/runs/detect/train/weights/best.pt")

# ✅ Run validation on test split
metrics = model.val(
    data="D:/ml2/data/data.yaml",
    split="test",
    conf=0.25
)

# ✅ Safely extract confusion matrix and class names
conf_matrix = getattr(metrics.confusion_matrix, "matrix", np.array([]))
class_names = getattr(metrics, "names", {})

if conf_matrix.size == 0 or np.sum(conf_matrix) == 0:
    print("⚠️ No detections found. Check your test labels or lower the confidence threshold.")
else:
    # ✅ Compute precision, recall, F1, accuracy
    precision = float(np.mean(getattr(metrics.box, "p", [0])))
    recall = float(np.mean(getattr(metrics.box, "r", [0])))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # ✅ Print metrics
    print("\n📊 Evaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"mAP50:     {getattr(metrics.box, 'map50', 0):.4f}")
    print(f"mAP50-95:  {getattr(metrics.box, 'map', 0):.4f}")

    # ✅ Print confusion matrix
    print("\n🧮 Confusion Matrix:")
    num_classes = conf_matrix.shape[0]
    header = [""] + [f"Pred_{class_names.get(i, f'Class_{i}')}" for i in range(num_classes)]
    rows = []
    for i, row in enumerate(conf_matrix):
        true_label = class_names.get(i, f"Class_{i}")
        rows.append([f"True_{true_label}"] + list(row))

    df = pd.DataFrame(rows, columns=header)
    print(df.to_string(index=False))