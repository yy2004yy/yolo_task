import os
import random
import cv2
import numpy as np
import yaml


def generate_complex_background(width, height):
    """生成复杂的随机背景"""
    # 随机选择背景类型：纯色、渐变、噪点或随机形状
    bg_type = random.choice(['gradient', 'noise', 'shapes', 'solid'])

    if bg_type == 'solid':
        # 纯色背景
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        background = np.full((height, width, 3), color, dtype=np.uint8)

    elif bg_type == 'gradient':
        # 渐变背景
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 创建渐变
        background = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):  # 对BGR三个通道分别处理
            if random.random() > 0.5:  # 50%概率水平渐变，50%概率垂直渐变
                background[:, :, i] = np.linspace(color1[i], color2[i], width, dtype=np.uint8)
            else:
                background[:, :, i] = np.linspace(color1[i], color2[i], height, dtype=np.uint8).reshape(-1, 1)

    elif bg_type == 'noise':
        # 噪点背景
        background = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # 添加模糊使噪点更自然
        background = cv2.GaussianBlur(background, (5, 5), 0)

    elif bg_type == 'shapes':
        # 随机形状背景
        background = np.zeros((height, width, 3), dtype=np.uint8)
        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        background[:, :] = base_color

        # 添加随机数量的形状
        num_shapes = random.randint(5, 20)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'circle', 'ellipse', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # 确保厚度是有效值
            if random.random() > 0.5:  # 50%概率填充
                thickness = -1
            else:
                thickness = random.randint(1, 3)  # 线宽1-3像素

            if shape_type == 'rectangle':
                pt1 = (random.randint(0, width), random.randint(0, height))
                pt2 = (random.randint(0, width), random.randint(0, height))
                cv2.rectangle(background, pt1, pt2, color, thickness)

            elif shape_type == 'circle':
                center = (random.randint(0, width), random.randint(0, height))
                radius = random.randint(10, min(width, height) // 4)
                cv2.circle(background, center, radius, color, thickness)

            elif shape_type == 'ellipse':
                center = (random.randint(0, width), random.randint(0, height))
                axes = (random.randint(10, width // 4), random.randint(10, height // 4))
                angle = random.randint(0, 180)
                start_angle = random.randint(0, 360)
                end_angle = random.randint(0, 360)
                cv2.ellipse(background, center, axes, angle, start_angle, end_angle, color, thickness)

            elif shape_type == 'line':
                pt1 = (random.randint(0, width), random.randint(0, height))
                pt2 = (random.randint(0, width), random.randint(0, height))
                # 确保线宽是有效值
                line_thickness = random.randint(1, 3)  # 线宽1-3像素
                cv2.line(background, pt1, pt2, color, line_thickness)

        # 添加一些噪点使形状更自然
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        background = cv2.add(background, noise)

    # 随机应用一些额外的效果
    if random.random() > 0.7:  # 30%概率添加模糊
        background = cv2.GaussianBlur(background, (5, 5), 0)

    if random.random() > 0.7:  # 30%概率调整亮度/对比度
        alpha = random.uniform(0.5, 1.5)
        beta = random.randint(-50, 50)
        background = cv2.convertScaleAbs(background, alpha=alpha, beta=beta)

    return background


def get_class_id(obj_file):
    """根据文件名返回类别ID"""
    obj_name = os.path.splitext(obj_file)[0].lower()
    if 'elephant' in obj_name:
        return 0
    elif 'monkey' in obj_name:
        return 1
    elif 'peacock' in obj_name:
        return 2
    elif 'tiger' in obj_name:
        return 3
    elif 'wolf' in obj_name:
        return 4
    else:
        return 5  # 默认类别


def create_yolo_dataset(objects_dir, output_dir, num_images=100,
                        max_objects_per_image=2, val_ratio=0.2,
                        bg_width=1920, bg_height=1080):
    """
    创建YOLO格式的数据集
    - 使用随机生成的复杂背景
    - 保持物体原始大小不变
    - 每张背景最多放置2个物体
    - 自动分割训练集和验证集
    - 生成train.txt和val.txt
    """
    # 检查目录是否存在
    if not os.path.exists(objects_dir):
        print(f"错误：物体目录不存在 {objects_dir}")
        return

    # 获取所有物体文件
    object_files = [f for f in os.listdir(objects_dir) if f.lower().endswith('.png')]

    if not object_files:
        print(f"错误：物体目录中没有PNG文件 {objects_dir}")
        return

    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # 存储生成的所有图片信息
    all_images = []

    # 为每张生成的图片处理
    for i in range(num_images):
        # 生成复杂随机背景
        bg_img = generate_complex_background(bg_width, bg_height)
        composite = bg_img.copy()

        # 创建一个空白掩码图，用于记录已放置物体的非透明区域
        occupied_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)

        # 决定要放置的物体数量 (1到max_objects_per_image)
        num_objects = random.randint(1, max_objects_per_image)
        annotations = []

        for _ in range(num_objects):
            # 随机选择物体（允许重复选择同一物体）
            obj_file = random.choice(object_files)
            obj_path = os.path.join(objects_dir, obj_file)
            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)

            if obj_img is None:
                print(f"警告：无法读取物体图片 {obj_path}，跳过")
                continue

            # 保持物体原始大小
            new_height, new_width = obj_img.shape[:2]

            # 确保物体不会超出背景边界
            if new_width > bg_width or new_height > bg_height:
                print(f"警告：物体 {obj_file} 尺寸超过背景大小，跳过")
                continue

            # 获取物体的非透明区域掩码（alpha > 0 的区域）
            if obj_img.shape[2] == 4:
                obj_mask = (obj_img[:, :, 3] > 0).astype(np.uint8)
            else:
                obj_mask = np.ones((new_height, new_width), dtype=np.uint8)

            # 随机位置 (确保物体完全在背景内)
            max_attempts = 20  # 增加尝试次数
            placed = False

            for _ in range(max_attempts):
                x = random.randint(0, bg_width - new_width)
                y = random.randint(0, bg_height - new_height)

                # 检查物体非透明区域是否与已放置物体的非透明区域重叠
                roi = occupied_mask[y:y + new_height, x:x + new_width]
                overlap = np.sum(roi * obj_mask) > 0

                if not overlap:
                    placed = True
                    break

            if not placed:
                print(f"警告：无法为物体 {obj_file} 找到合适位置，跳过")
                continue

            # 更新已占用区域掩码
            occupied_mask[y:y + new_height, x:x + new_width] = np.maximum(
                occupied_mask[y:y + new_height, x:x + new_width],
                obj_mask
            )

            # 合成物体到背景
            if obj_img.shape[2] == 4:
                obj_rgb = obj_img[:, :, :3]
                alpha = obj_img[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)

                roi = composite[y:y + new_height, x:x + new_width]
                composite[y:y + new_height, x:x + new_width] = (1 - alpha) * roi + alpha * obj_rgb
            else:
                composite[y:y + new_height, x:x + new_width] = obj_img[:, :, :3]

            # 计算YOLO格式的标注 (class x_center y_center width height)
            x_center = (x + new_width / 2) / bg_width
            y_center = (y + new_height / 2) / bg_height
            width = new_width / bg_width
            height = new_height / bg_height

            # 类别ID
            class_id = get_class_id(obj_file)

            # 添加到标注列表
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存图片和标注信息
        img_filename = f"image_{i:04d}.jpg"
        all_images.append(img_filename)

        # 保存合成图片（临时位置，稍后移动）
        temp_img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(temp_img_path, composite)

        # 保存标注文件（临时位置，稍后移动）
        label_filename = f"image_{i:04d}.txt"
        temp_label_path = os.path.join(output_dir, label_filename)
        with open(temp_label_path, 'w') as f:
            for annotation in annotations:
                f.write(annotation + "\n")

        print(f"已生成: {img_filename} 包含 {len(annotations)} 个物体")

        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'elephant', 1: 'monkey', 2: 'Peacock', 3: 'tiger', 4: 'wolf'},
            'nc': 5
        }

        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

    # 分割训练集和验证集
    random.shuffle(all_images)  # 随机打乱
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_names = all_images[:split_idx]
    val_names = all_images[split_idx:]

    # 移动文件到对应的目录
    for img_name in train_names:
        # 移动图片
        src_img = os.path.join(output_dir, img_name)
        dst_img = os.path.join(output_dir, "images", "train", img_name)
        os.rename(src_img, dst_img)

        # 移动标签
        label_name = img_name.replace(".jpg", ".txt")
        src_label = os.path.join(output_dir, label_name)
        dst_label = os.path.join(output_dir, "labels", "train", label_name)
        os.rename(src_label, dst_label)

    for img_name in val_names:
        # 移动图片
        src_img = os.path.join(output_dir, img_name)
        dst_img = os.path.join(output_dir, "images", "val", img_name)
        os.rename(src_img, dst_img)

        # 移动标签
        label_name = img_name.replace(".jpg", ".txt")
        src_label = os.path.join(output_dir, label_name)
        dst_label = os.path.join(output_dir, "labels", "val", label_name)
        os.rename(src_label, dst_label)

    # 生成train.txt和val.txt
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        for img_name in train_names:
            img_path = os.path.join("images", "train", img_name)
            f.write(img_path + "\n")

    with open(os.path.join(output_dir, "val.txt"), 'w') as f:
        for img_name in val_names:
            img_path = os.path.join("images", "val", img_name)
            f.write(img_path + "\n")

    print(f"\n数据集生成完成！")
    print(f"训练集数量: {len(train_names)}")
    print(f"验证集数量: {len(val_names)}")
    print(f"训练集列表: {os.path.join(output_dir, 'train.txt')}")
    print(f"验证集列表: {os.path.join(output_dir, 'val.txt')}")


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置目录路径
    objects_dir = os.path.join(script_dir, "objects")
    output_directory = os.path.join(script_dir, "yolo_dataset_color")

    # 创建必要的目录
    os.makedirs(objects_dir, exist_ok=True)

    # 参数设置
    create_yolo_dataset(
        objects_dir=objects_dir,
        output_dir=output_directory,
        num_images=100,
        max_objects_per_image=2,
        val_ratio=0.2,
        bg_width=1920,  # 背景宽度
        bg_height=1080  # 背景高度
    )