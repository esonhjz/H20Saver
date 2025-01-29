import os
import yaml
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------------
# 1. 初始化配置
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本的绝对路径
data_yaml_path = os.path.join(script_dir, "dataset", "data.yaml")  # 动态生成 data.yaml 的路径

if not os.path.exists(data_yaml_path):
    print(f"错误: data.yaml 文件不存在 - {data_yaml_path}")
    exit()

with open(data_yaml_path, "r") as f:
    data_config = yaml.safe_load(f)  # 加载配置文件
class_names = data_config["names"]  # 从 data.yaml 获取类别名称
num_classes = len(class_names)  # 类别数量
num_keypoints = 17  # 假设姿态关键点数量

# 数据集路径
dataset_dir = os.path.join(script_dir, "dataset")  # 数据集根目录
train_dir = os.path.join(dataset_dir, data_config["train"])
val_dir = os.path.join(dataset_dir, data_config["val"])
test_dir = os.path.join(dataset_dir, data_config["test"])

# 确保模型文件存在
model_path = "yolo11x-pose.pt"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在 - {model_path}")
    exit()

# 模型加载
device = "cpu"  # 强制使用CPU
yolo_model = YOLO(model_path)  # 直接加载模型权重文件

# 初始化未检测到关键点的计数器
no_keypoints_count = 0

# ----------------------------
# 2. 遍历数据集并进行姿态关键点检测
# ----------------------------
for dataset_dir in [train_dir, val_dir, test_dir]:
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    keypoints_label_dir = os.path.join(dataset_dir, "keypoints_labels")  # 创建每个数据集的关键点标签目录

    os.makedirs(keypoints_label_dir, exist_ok=True)  # 确保目录存在

    # 遍历图像文件
    for img_file in os.listdir(image_dir):
        if not img_file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 进行姿态关键点检测
        results = yolo_model.predict(img, conf=0.25, iou=0.7)

        keypoints = []
        has_valid_keypoints = False  # 标记是否有有效关键点

        for result in results:
            if result.keypoints is not None:
                keypoints.extend(result.keypoints.xy)  # 获取关键点坐标
                # 检查是否有有效关键点
                valid_keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 2)
                has_valid_keypoints = any(
                    (-1 <= kpt[0] <= 1) and (-1 <= kpt[1] <= 1)
                    for kpt in valid_keypoints
                )
                if has_valid_keypoints:
                    break  # 至少有一个有效关键点即可

        if has_valid_keypoints:
            # 保存关键点到标签文件
            output_label_file = os.path.join(keypoints_label_dir, label_file)
            with open(output_label_file, 'w') as f:
                for kpt in keypoints:
                    f.write(f"{kpt[0]} {kpt[1]}\n")
        else:
            # 如果没有任何有效关键点，增加计数器
            no_keypoints_count += 1
            print(f"Warning: No keypoints detected for {img_file}")

# ----------------------------
# 3. 更新 YAML 配置文件
# ----------------------------
config_path = data_yaml_path
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['keypoints'] = {
    'num_keypoints': num_keypoints,
    'keypoint_labels': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
}

with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)

# ----------------------------
# 4. 输出结果
# ----------------------------
print(f"姿态关键点预测完成，结果分别保存在 train, val, test 数据集下的 keypoints_labels 文件夹中")
print(f"未检测到任何关键点的图像数量: {no_keypoints_count}")