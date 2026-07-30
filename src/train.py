import yaml
import os
import numpy as np
from ultralytics import YOLO
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# 配置参数
# ----------------------------
class CustomConfig:
    # 数据参数
    data_yaml = "dataset/data.yaml"
    img_size = 640
    batch_size = 23
    epochs = 100
    patience = 10

    # 数据增强参数
    augment = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.9,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
    }

    # 优化参数
    lr0 = 0.001
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005

    # 损失函数权重 (Ultralytics 原生参数: box / cls / dfl)
    # cls 权重从默认 0.5 提高到 1.0 以应对类别不平衡
    loss_box_gain: float = 7.5
    loss_cls_gain: float = 1.0
    loss_dfl_gain: float = 1.5

    # 模型参数
    pretrained = "yolo11x.pt"
    freeze_layers = None       # list[int] | None, 如 [0, 1, 2] 冻结前 3 层
    multi_scale: bool = True

    # 训练输出
    project: str = "runs/detect"
    experiment_name: str = "h20saver"
    resume: bool = False


# ----------------------------
# 数据准备：分析类别分布
# ----------------------------
def prepare_dataset(config):
    with open(config.data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_labels = []
    train_label_dir = os.path.join(
        os.path.dirname(config.data_yaml),
        data["train"].replace("images", "labels"),
    )
    for label_file in os.listdir(train_label_dir):
        label_path = os.path.join(train_label_dir, label_file)
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                    train_labels.append(class_id)
                except (ValueError, IndexError):
                    continue

    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = {int(k): round(float(v), 3) for k, v in zip(classes, weights)}

    print(f"[数据] 类别权重 (仅供参考): {class_weights}")
    print("[数据] 注意: Ultralytics YOLO 不支持直接注入逐类别权重。")
    print("[数据] 若类别不平衡严重, 请通过数据增强或过采样补充少数类。")
    return class_weights


# ----------------------------
# 模型训练
# ----------------------------
def train_yolo(config):
    model = YOLO(config.pretrained)

    train_kwargs = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.img_size,
        "batch": config.batch_size,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "patience": config.patience,
        "optimizer": "AdamW",
        "cos_lr": True,
        "freeze": config.freeze_layers,
        "multi_scale": config.multi_scale,
        "project": config.project,
        "name": config.experiment_name,
        "resume": config.resume,
        # 数据增强
        "hsv_h": config.augment["hsv_h"],
        "hsv_s": config.augment["hsv_s"],
        "hsv_v": config.augment["hsv_v"],
        "translate": config.augment["translate"],
        "scale": config.augment["scale"],
        "fliplr": config.augment["fliplr"],
        "mosaic": config.augment["mosaic"],
        "mixup": config.augment["mixup"],
        # 损失函数增益 — YOLO11 内置 DFL，cls 增益提高缓解类别不平衡
        "box": config.loss_box_gain,
        "cls": config.loss_cls_gain,
        "dfl": config.loss_dfl_gain,
    }

    print("=" * 40)
    print("  开始模型训练...")
    print("=" * 40)

    results = model.train(**train_kwargs)

    final_weights = f"{config.experiment_name}_final.pt"
    model.save(final_weights)
    print(f"[权重] 已保存至: {final_weights}")

    return model


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    config = CustomConfig()
    class_weights = prepare_dataset(config)
    model = train_yolo(config)

    # 导出 ONNX (FP16 半精度简化推理)
    print("\n[导出] 正在导出 ONNX 模型...")
    model.export(format="onnx", imgsz=config.img_size, half=True, simplify=True)

    # 验证
    print("\n[验证] 正在评估模型...")
    metrics = model.val()
    print(
        f"验证结果: mAP@0.5={metrics.box.map:.3f}, "
        f"mAP@0.5:0.95={metrics.box.map50:.3f}"
    )