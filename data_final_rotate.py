import os
import random
import cv2
import numpy as np
import yaml
from math import radians, cos, sin


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


def rotate_image(image, angle):
    """旋转图像并返回旋转后的图像及其掩码"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新的边界尺寸
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # 调整旋转矩阵
    rot_mat[0, 2] += new_w / 2 - center[0]
    rot_mat[1, 2] += new_h / 2 - center[1]

    # 执行旋转
    flags = cv2.INTER_LINEAR
    rotated = cv2.warpAffine(
        image, rot_mat, (new_w, new_h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    return rotated


def get_rotated_bbox(mask):
    """从旋转后的掩码计算水平外接矩形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 合并所有轮廓
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return x, y, w, h


def create_yolo_dataset_with_rotation(
        backgrounds_dir,
        objects_dir,
        output_dir,
        num_images=100,
        max_objects_per_image=2,
        val_ratio=0.2,
        max_rotation=15
):
    """
    创建带旋转增强的标准YOLO数据集
    - 物体会有随机旋转但标注使用水平矩形框
    - 旋转角度范围: -max_rotation到+max_rotation度
    """
    # 检查目录
    if not all(os.path.exists(d) for d in [backgrounds_dir, objects_dir]):
        print("错误：背景或物体目录不存在")
        return

    # 获取文件列表
    bg_files = [f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    obj_files = [f for f in os.listdir(objects_dir) if f.lower().endswith('.png')]

    if not bg_files or not obj_files:
        print("错误：缺少背景或物体文件")
        return

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    all_images = []

    for i in range(num_images):
        # 选择背景
        bg_path = os.path.join(backgrounds_dir, random.choice(bg_files))
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]
        composite = bg_img.copy()
        occupied_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        annotations = []

        # 随机物体数量
        num_objs = random.randint(1, max_objects_per_image)

        for _ in range(num_objs):
            # 选择物体并读取
            obj_file = random.choice(obj_files)
            obj_path = os.path.join(objects_dir, obj_file)
            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img is None:
                continue

            # 添加随机旋转
            angle = random.uniform(-max_rotation, max_rotation)
            rotated_obj = rotate_image(obj_img, angle)
            rot_h, rot_w = rotated_obj.shape[:2]

            # 跳过过大的物体
            if rot_w > bg_w * 0.8 or rot_h > bg_h * 0.8:
                continue

            # 获取旋转后的掩码
            if rotated_obj.shape[2] == 4:
                obj_mask = (rotated_obj[:, :, 3] > 0).astype(np.uint8)
            else:
                obj_mask = np.ones((rot_h, rot_w), dtype=np.uint8)

            # 寻找合适位置
            placed = False
            for _ in range(20):  # 最多尝试20次
                x = random.randint(0, bg_w - rot_w)
                y = random.randint(0, bg_h - rot_h)

                # 检查重叠
                roi = occupied_mask[y:y + rot_h, x:x + rot_w]
                if np.sum(roi * obj_mask) == 0:
                    placed = True
                    break

            if not placed:
                continue

            # 更新占用区域
            occupied_mask[y:y + rot_h, x:x + rot_w] = np.maximum(
                occupied_mask[y:y + rot_h, x:x + rot_w],
                obj_mask
            )

            # 合成物体
            if rotated_obj.shape[2] == 4:
                obj_rgb = rotated_obj[:, :, :3]
                alpha = rotated_obj[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                composite[y:y + rot_h, x:x + rot_w] = \
                    (1 - alpha) * composite[y:y + rot_h, x:x + rot_w] + alpha * obj_rgb
            else:
                composite[y:y + rot_h, x:x + rot_w] = rotated_obj[:, :, :3]

            # 计算水平外接矩形
            full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            full_mask[y:y + rot_h, x:x + rot_w] = obj_mask
            x_rect, y_rect, w_rect, h_rect = get_rotated_bbox(full_mask)

            # 转换为YOLO格式
            x_center = (x_rect + w_rect / 2) / bg_w
            y_center = (y_rect + h_rect / 2) / bg_h
            width = w_rect / bg_w
            height = h_rect / bg_h

            class_id = get_class_id(obj_file)
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存结果
        img_name = f"image_{i:04d}.jpg"
        all_images.append(img_name)

        cv2.imwrite(os.path.join(output_dir, img_name), composite)

        with open(os.path.join(output_dir, img_name.replace(".jpg", ".txt")), 'w') as f:
            f.write("\n".join(annotations))

        print(f"生成: {img_name} (包含 {len(annotations)} 个物体)")

    # 创建data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'elephant', 1: 'monkey', 2: 'peacock', 3: 'tiger', 4: 'wolf'},
        'nc': 5
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    # 分割数据集
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    # 移动文件到对应目录
    for f in train_files:
        os.rename(
            os.path.join(output_dir, f),
            os.path.join(output_dir, "images", "train", f)
        )
        os.rename(
            os.path.join(output_dir, f.replace(".jpg", ".txt")),
            os.path.join(output_dir, "labels", "train", f.replace(".jpg", ".txt"))
        )

    for f in val_files:
        os.rename(
            os.path.join(output_dir, f),
            os.path.join(output_dir, "images", "val", f)
        )
        os.rename(
            os.path.join(output_dir, f.replace(".jpg", ".txt")),
            os.path.join(output_dir, "labels", "val", f.replace(".jpg", ".txt"))
        )

    print(f"\n数据集生成完成！共 {len(all_images)} 张图片")
    print(f"训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")


if __name__ == "__main__":
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backgrounds_dir = os.path.join(script_dir, "backgrounds")
    objects_dir = os.path.join(script_dir, "objects")
    output_dir = os.path.join(script_dir, "yolo_dataset_rotated")

    # 运行
    create_yolo_dataset_with_rotation(
        backgrounds_dir=backgrounds_dir,
        objects_dir=objects_dir,
        output_dir=output_dir,
        num_images=100,
        max_objects_per_image=2,
        val_ratio=0.2,
        max_rotation=15  # 控制旋转幅度
    )