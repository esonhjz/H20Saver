import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


# ----------------------------
# 初始化配置
# ----------------------------

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_analysis.out")
os.makedirs(output_dir, exist_ok=True)

# 初始化统计计数器
stats = {
    "total_images": 0,          # 总图像数量
    "missing_labels": 0,        # 缺失标注文件数量
    "corrupted_images": 0,      # 损坏/不可读图像数量
    "invalid_annotations": 0    # 无效标注行数
}

# ----------------------------
# 加载数据集配置
# ----------------------------
# 定位 data.yaml 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(script_dir, "dataset", "data.yaml")

if not os.path.exists(data_yaml_path):
    print(f"错误: data.yaml 文件不存在 - {data_yaml_path}")
    exit()

# 读取配置文件
with open(data_yaml_path, "r") as file:
    data = yaml.safe_load(file)

# 动态构建路径
base_dir = os.path.dirname(data_yaml_path)  # 数据集根目录
class_names = data["names"]

# 构建数据集路径
train_dir = os.path.join(base_dir, data["train"])
val_dir = os.path.join(base_dir, data["val"])
test_dir = os.path.join(base_dir, data["test"])

# 构建标注目录
train_label_dir = os.path.join(base_dir, "train/labels")
val_label_dir = os.path.join(base_dir, "valid/labels")
test_label_dir = os.path.join(base_dir, "test/labels")

# 检查关键目录是否存在
for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        print(f"错误：目录不存在 - {path}")
        exit()

# ----------------------------
# 1. 统计类别分布
# ----------------------------
def count_class_distribution(image_dir, label_dir):
    image_dir = os.path.join(image_dir, "images")
    class_counts = defaultdict(int)
    
    # 遍历图像文件检查标注
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    stats["total_images"] += len(image_files)  # 统计总图像数
    
    for img_file in image_files:
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            stats["missing_labels"] += 1
            continue  # 跳过缺失标注的文件

    # 统计有效标注
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                    except:
                        stats["invalid_annotations"] += 1
        except Exception as e:
            print(f"读取标注文件失败: {label_path} - {str(e)}")
    
    return class_counts

# 统计各数据集分布
train_counts = count_class_distribution(train_dir, train_label_dir)
val_counts = count_class_distribution(val_dir, val_label_dir)
test_counts = count_class_distribution(test_dir, test_label_dir)

# 绘制类别分布图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names))
width = 0.25

ax.bar(x - width, [train_counts[i] for i in range(len(class_names))], width, label="训练集")
ax.bar(x, [val_counts[i] for i in range(len(class_names))], width, label="验证集")
ax.bar(x + width, [test_counts[i] for i in range(len(class_names))], width, label="测试集")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45)
ax.set_ylabel("样本数量")
ax.set_title("数据集的类别分布")
ax.legend()

plt.savefig(os.path.join(output_dir, "class_distribution.png"), bbox_inches="tight")
plt.close()


# ----------------------------
# 2. 标注框分析
# ----------------------------
def analyze_bbox(label_dir):
    widths, heights, centers_x, centers_y = [], [], [], []
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
            
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f.readlines():
                try:
                    _, x, y, w, h = map(float, line.strip().split())
                    widths.append(w)
                    heights.append(h)
                    centers_x.append(x)
                    centers_y.append(y)
                except:
                    stats["invalid_annotations"] += 1
    
    return widths, heights, centers_x, centers_y

# ----------------------------
# 3. 绘制分析图
# ----------------------------

widths, heights, centers_x, centers_y = analyze_bbox(train_label_dir)

plt.figure(figsize=(15, 6))

# 尺寸分布
plt.subplot(1, 2, 1)
plt.hist([widths, heights], bins=50, label=["宽度", "高度"], alpha=0.7)
plt.xlabel("归一化尺寸值")
plt.ylabel("频次")
plt.title("标注框尺寸分布")
plt.legend()

# 中心点分布
plt.subplot(1, 2, 2)
plt.scatter(centers_x, centers_y, s=5, alpha=0.3)
plt.xlabel("归一化X坐标")
plt.ylabel("归一化Y坐标")
plt.title("标注框中心点分布")

plt.savefig(os.path.join(output_dir, "bbox_analysis.png"), bbox_inches="tight")
plt.close()

# 饼图：类别分布
plt.figure(figsize=(8, 8))
plt.pie(
    [train_counts[i] for i in range(len(class_names))],
    labels=class_names,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("训练集类别分布")
plt.savefig(os.path.join(output_dir, "class_distribution_pie.png"), bbox_inches="tight")
plt.close()

# 热力图：标注框中心点分布
plt.figure(figsize=(10, 8))
sns.kdeplot(x=centers_x, y=centers_y, cmap="Reds", shade=True)
plt.xlabel("归一化X坐标")
plt.ylabel("归一化Y坐标")
plt.title("标注框中心点热力图")
plt.savefig(os.path.join(output_dir, "bbox_heatmap.png"), bbox_inches="tight")
plt.close()

# 箱线图：标注框尺寸分布
plt.figure(figsize=(10, 6))
plt.boxplot([widths, heights], labels=["宽度", "高度"])
plt.xlabel("标注框尺寸")
plt.ylabel("归一化值")
plt.title("标注框尺寸箱线图")
plt.savefig(os.path.join(output_dir, "bbox_boxplot.png"), bbox_inches="tight")
plt.close()

# 条形图：类别不平衡分析
plt.figure(figsize=(10, 6))
plt.bar(class_names, [train_counts[i] for i in range(len(class_names))])
plt.xlabel("类别")
plt.ylabel("样本数量")
plt.title("训练集类别样本数量")
plt.savefig(os.path.join(output_dir, "class_imbalance.png"), bbox_inches="tight")
plt.close()

# ----------------------------
# 生成数据质量报告
# ----------------------------
if stats['total_images'] > 0:
    missing_percent = stats['missing_labels'] / stats['total_images']
    corrupted_percent = stats['corrupted_images'] / stats['total_images']
else:
    missing_percent = 0.0
    corrupted_percent = 0.0

report = f"""
{'='*20} 数据质量报告 {'='*20}
总图像数量: {stats['total_images']}
有效标注图像: {stats['total_images'] - stats['missing_labels'] - stats['corrupted_images']}
缺失标注文件: {stats['missing_labels']} ({missing_percent:.1%})
损坏/不可读图像: {stats['corrupted_images']} ({corrupted_percent:.1%})
无效标注行数: {stats['invalid_annotations']}
{'='*40}
"""

print(report)

# 保存报告到文件
report_file_path = os.path.join(output_dir, "data_quality_report.txt")
with open(report_file_path, "w", encoding="utf-8") as report_file:
    report_file.write(report)