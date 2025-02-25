import torch
import yaml
import os
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F
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
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.9,
        'fliplr': 0.5,
        'mosaic': 1.0,  # 增加 Mosaic 增强的概率
        'mixup': 0.2     # 增加 Mixup 的概率
    }
    
    # 优化参数
    lr0 = 0.001    # 初始学习率
    lrf = 0.01    # 最终学习率
    momentum = 0.937
    weight_decay = 0.0005
    
    # 类别平衡参数
    class_weights = None  # 自动计算
    focal_loss = True     # 启用焦点损失
    
    # 模型参数
    pretrained = "yolo11x.pt"
    freeze = ['backbone', 'head']  # 冻结的层
    multi_scale = True    # 多尺度训练

    # IoU 阈值恢复为 0.5
    iou_threshold = 0.5  # IoU阈值设置回0.5

# ----------------------------
# 数据准备与类别平衡
# ----------------------------
def prepare_dataset(config):
    # 分析数据集分布
    with open(config.data_yaml) as f:
        data = yaml.safe_load(f)
    
    # 计算类别权重
    train_labels = []
    train_label_dir = os.path.join(os.path.dirname(config.data_yaml), data['train'].replace('images', 'labels'))
    for label_file in os.listdir(train_label_dir):
        with open(os.path.join(train_label_dir, label_file)) as f:
            for line in f.readlines():
                class_id = int(line.strip().split()[0])
                train_labels.append(class_id)
    
    classes = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    config.class_weights = {k:v for k,v in zip(classes, weights)}
    
    print(f"Class weights calculated: {config.class_weights}")

# ----------------------------
# 自定义损失函数
# ----------------------------
class CustomLoss:
    def __init__(self, model, class_weights=None, focal_gamma=2.0):
        self.model = model
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
    
    def __call__(self, preds, targets):
        # 修改默认损失计算
        loss = self.model.compute_loss(preds, targets)
        
        # 应用类别权重
        if self.class_weights:
            cls_loss = loss[1]  # 分类损失分量
            weighted_cls_loss = cls_loss * self.class_weights
            loss[1] = weighted_cls_loss.mean()
        
        # 应用焦点损失
        if CustomConfig.focal_loss:
            p = torch.sigmoid(preds[..., 4:])
            ce_loss = F.binary_cross_entropy_with_logits(p, targets, reduction='none')
            alpha = torch.where(targets==1, 0.25, 0.75)  # 自定义alpha参数
            focal_loss = alpha * (1 - p)**self.focal_gamma * ce_loss
            loss[1] += focal_loss.mean()
        
        return loss.sum()

# ----------------------------
# 模型训练
# ----------------------------

def train_yolo(config):
    # 初始化模型
    model_path = '/root/.cache/torch/hub/checkpoints/yolo11x.pt'  # 指定缓存路径
    model = YOLO(model_path)  # 使用本地路径加载模型
    model.loss = CustomLoss(model, class_weights=config.class_weights)

    # 冻结指定层
    if config.freeze:
        freeze = [f'model.{x}' for x in config.freeze]
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                v.requires_grad = False

    # 设置优化器 (AdamW 或其他)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr0, weight_decay=config.weight_decay)

    # 学习率调度器：Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 训练参数
    train_args = {
        'data': config.data_yaml,
        'epochs': config.epochs,
        'imgsz': config.img_size,
        'batch': config.batch_size,
        'lr0': config.lr0,
        'lrf': config.lrf,
        'momentum': config.momentum,
        'weight_decay': config.weight_decay,
        'patience': config.patience,
        'augment': True,
        **config.augment
    }

    # 检查是否存在检查点
    checkpoint_path = "yolov11_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"从第 {start_epoch} 轮开始继续训练")

    # 开始训练
    print('''
          ====================
          开始模型训练...
          ====================
          ''')
    for epoch in range(start_epoch, config.epochs):
        # 每个epoch开始时更新学习率
        scheduler.step()

        # 训练过程代码...
        results = model.train(**train_args)

        # 保存检查点
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
        }, checkpoint_path)

    return model
# ----------------------------
# 主程序
# ----------------------------
if __name__ == '__main__':
    config = CustomConfig()
    prepare_dataset(config)
    model = train_yolo(config)
    
    # 导出模型
    model.export(format='onnx', imgsz=config.img_size)
    
    # 验证结果
    metrics = model.val()
    print(f"验证结果:mAP@0.5={metrics.box.map:.2f}, mAP@0.5:0.95={metrics.box.map50:.2f}")
