import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# 配置参数
# ----------------------------
CONF_THRESH = 0.25  # 置信度阈值
IOU_THRESH = 0.5    # IoU阈值
NUM_IMAGES = 2     # 选择图片数量

# ----------------------------
# 初始化模型和数据集
# ----------------------------
model = YOLO("yolo11n.pt")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(script_dir, "dataset", "data.yaml")

with open(data_yaml_path, "r") as f:
    data = yaml.safe_load(f)
    
base_dir = os.path.dirname(data_yaml_path)
test_dir = os.path.join(base_dir, data["test"], "images")
test_label_dir = os.path.join(base_dir, "test/labels")
class_names = data["names"]

# ----------------------------
# 获取测试图片列表
# ----------------------------
test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_images = np.random.choice(test_images, min(NUM_IMAGES, len(test_images)), replace=False)


# ----------------------------
# 评估指标容器
# ----------------------------
metrics = {
    "total": defaultdict(int),
    "classes": defaultdict(lambda: {
        "TP": 0, "FP": 0, "FN": 0,
        "precision": 0, "recall": 0, "f1": 0
    })
}

# ----------------------------
# 处理函数
# ----------------------------
def parse_label(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, xc, yc, w, h = map(float, line.strip().split())
            x1 = (xc - w/2) * img_width
            y1 = (yc - h/2) * img_height
            x2 = (xc + w/2) * img_width
            y2 = (yc + h/2) * img_height
            boxes.append([int(class_id), x1, y1, x2, y2])
    return boxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# ----------------------------
# 主处理流程
# ----------------------------
for img_name in tqdm(selected_images, desc="Processing Images"):
    # 读取图片
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 获取对应标签
    label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + ".txt")
    gt_boxes = parse_label(label_path, w, h) if os.path.exists(label_path) else []
    
    # 模型预测
    results = model.predict(img, conf=CONF_THRESH)
    detections = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            class_id = int(box.cls)
            conf = float(box.conf)
            detections.append([class_id, conf, *xyxy])
    
    # 匹配检测结果
    matched_gt = set()
    for det in sorted(detections, key=lambda x: -x[1]):
        max_iou = 0
        match_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = calculate_iou(det[2:], gt[1:])
            if iou > max_iou and det[0] == gt[0]:
                max_iou = iou
                match_idx = i
                
        if max_iou >= IOU_THRESH:
            metrics["classes"][det[0]]["TP"] += 1
            metrics["total"]["TP"] += 1
            matched_gt.add(match_idx)
        else:
            metrics["classes"][det[0]]["FP"] += 1
            metrics["total"]["FP"] += 1
    
    # 统计漏检
    metrics["total"]["FN"] += len(gt_boxes) - len(matched_gt)
    for gt in gt_boxes:
        if gt[0] not in [det[0] for det in detections]:
            metrics["classes"][gt[0]]["FN"] += 1

# ----------------------------
# 计算指标
# ----------------------------
def calculate_metrics(stats):
    precision = stats["TP"] / (stats["TP"] + stats["FP"] + 1e-6)
    recall = stats["TP"] / (stats["TP"] + stats["FN"] + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

# 总体指标
total_precision, total_recall, total_f1 = calculate_metrics(metrics["total"])
metrics["total"].update({
    "precision": total_precision,
    "recall": total_recall,
    "f1": total_f1
})

# 分类指标
for class_id in metrics["classes"]:
    stats = metrics["classes"][class_id]
    precision, recall, f1 = calculate_metrics(stats)
    stats.update({
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# ----------------------------
# 生成报告
# ----------------------------
print("\n\n评估报告: ")
print(f"测试图片数量: {len(selected_images)}")
print(f"总检测数: {metrics['total']['TP'] + metrics['total']['FP']}")
print(f"总真实标注数: {metrics['total']['TP'] + metrics['total']['FN']}")
print(f"全局精确率: {metrics['total']['precision']:.2%}")
print(f"全局召回率: {metrics['total']['recall']:.2%}")
print(f"全局F1分数: {metrics['total']['f1']:.2%}")

print("\n分类别统计:")
print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
for class_id in sorted(metrics["classes"]):
    cls = metrics["classes"][class_id]
    name = class_names[class_id]
    print(f"{name:<15} {cls['precision']:.2%} {cls['recall']:.2%} {cls['f1']:.2%} "
          f"{cls['TP']:<5} {cls['FP']:<5} {cls['FN']:<5}")
    

