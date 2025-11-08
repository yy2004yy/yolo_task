import os
from ultralytics import YOLO

def train():
    # 硬编码配置（直接替换您的实际路径）
    PRETRAINED_MODEL = 'runs/weights/best.pt'  # 替换为您的生成数据模型
    DATASET_YAML = 'yolo_dataset_color/data.yaml'

    # 检查路径
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"数据集配置文件 {DATASET_YAML} 不存在！")
    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(f"预训练模型 {PRETRAINED_MODEL} 不存在！")

    # 加载您的生成数据预训练模型
    model = YOLO(PRETRAINED_MODEL)

    # 数据微调配置
    train_args = {
        'data': DATASET_YAML,
        'epochs': 150,
        'batch': 4,
        'imgsz': 640,
        'lr0': 0.001,  # 初始学习率
        'device': '3',  # 使用GPU3，若使用CPU，则使用代码：'device': 'cpu'
        'workers': 4,
        'augment': True,  # 保持基础增强
        'freeze': 10,  # 冻结前10层骨干网络
        'patience': 30,  # 早停轮次
        'dropout': 0.1,  # 防过拟合
        'optimizer': 'AdamW'
    }

    # 开始训练
    model.train(**train_args)

    # 验证（使用实拍验证集）
    model.val()


if __name__ == '__main__':
    train()