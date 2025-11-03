import os
import random
import cv2
import numpy as np
import yaml
from math import radians, cos, sin
import albumentations as A


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


def create_color_light_augmentation():
    """创建专注于颜色和光照变化的增强管道
    返回:
        A.Compose: 包含颜色和光照增强操作的组合管道
    """
    return A.Compose([
        # ==================== 颜色变换 ====================
        # HSV颜色空间变换
        A.HueSaturationValue(
            hue_shift_limit=0,  # 色调偏移范围[-20,20]度 (0-180)
            sat_shift_limit=40,  # 饱和度偏移范围[-30,30]
            val_shift_limit=20,  # 明度偏移范围[-20,20]
            p=0.7  # 70%概率应用此变换
        ),

        # 随机亮度对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # 亮度调整范围[-0.3,0.3] (0-1范围)
            contrast_limit=0.2,  # 对比度调整范围[-0.2,0.2]
            p=0.7  # 70%概率应用
        ),

        # 对比度受限的自适应直方图均衡化(CLAHE)
        A.CLAHE(
            clip_limit=3.0,  # 对比度限制阈值(越高对比度越强)
            tile_grid_size=(8, 8),  # 分块大小(8x8像素)
            p=0.3  # 30%概率应用
        ),

        # 过曝效果(模拟强光照射)
        A.Solarize(
            threshold=200,  # 像素值超过200时会反转(0-255范围)
            p=0.1  # 10%概率应用
        ),

        # 随机阴影效果
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),  # 阴影可出现在图像任何区域
            shadow_dimension=3,  # 阴影边缘模糊程度
            p=0.2  # 20%概率应用
        ),

        # 随机眩光效果(模拟阳光直射镜头)
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),  # 眩光可出现在图像任何区域
            angle_lower=0.5,  # 光源最小角度(弧度)
            angle_upper=1.5,  # 光源最大角度(弧度)
            num_flare_circles_lower=1,  # 最少光斑数量
            num_flare_circles_upper=6,  # 最多光斑数量
            src_radius=50,  # 光源半径(像素)
            src_color=(200, 200, 200),  # 光源颜色(RGB值)
            p=0.3  # 30%概率应用
        ),

        A.GaussianBlur(
            blur_limit=(3, 7),  # 高斯核大小 (奇数)
            sigma_limit=(0.1, 2.0),  # 自动计算sigma值
            p=0.25  # 应用概率25%
        ),
        A.Downscale(
            scale_min=0.5,  # 最小缩放比例 (0.25-1.0)
            scale_max=0.9,  # 最大缩放比例
            interpolation=cv2.INTER_LINEAR,  # 插值方法
            p=0.2  # 应用概率20%
        ),
    ])


def create_shake_augmentation():
    """创建背景摇晃(相机抖动)增强管道"""
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05,  # 轻微平移(5%)
            scale_limit=0.0,  # 不缩放
            rotate_limit=2,  # 轻微旋转(±2度)
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.OpticalDistortion(
            distort_limit=0.05,  # 轻微畸变
            shift_limit=0.0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        )
    ])


def rotate_image(image, angle):
    """旋转图像并返回旋转后的图像及其掩码"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    rot_mat[0, 2] += new_w / 2 - center[0]
    rot_mat[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(
        image, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    return rotated


def get_rotated_bbox(mask):
    """从旋转后的掩码计算水平外接矩形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return x, y, w, h


def apply_augmentations(image, augmentation_pipeline):
    """应用增强管道到图像"""
    if image.shape[2] == 4:  # 带alpha通道的图像
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]

        augmented = augmentation_pipeline(image=rgb)['image']

        if augmented.shape != rgb.shape:
            augmented = cv2.resize(augmented, (rgb.shape[1], rgb.shape[0]))
        augmented_image = np.dstack((augmented, alpha))
        return augmented_image
    else:
        augmented = augmentation_pipeline(image=image)['image']
        return augmented


def create_yolo_dataset_with_rotation(
        backgrounds_dir,
        objects_dir,
        output_dir,
        num_images=100,
        max_objects_per_image=2,
        val_ratio=0.2,
        max_rotation=15,
        min_scale=0.5,
        max_scale=1.2
):
    """
    创建带旋转增强、颜色/光照噪声和背景摇晃的标准YOLO数据集
    参数:
        backgrounds_dir: 背景图片目录
        objects_dir: 物体图片目录(带透明通道)
        output_dir: 输出目录
        num_images: 生成图片数量
        max_objects_per_image: 每张图片最多物体数
        val_ratio: 验证集比例
        max_rotation: 最大旋转角度(度)
        min_scale: 最小缩放比例
        max_scale: 最大缩放比例
    """
    if not all(os.path.exists(d) for d in [backgrounds_dir, objects_dir]):
        print("错误：背景或物体目录不存在")
        return

    bg_files = [f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    obj_files = [f for f in os.listdir(objects_dir) if f.lower().endswith('.png')]

    if not bg_files or not obj_files:
        print("错误：缺少背景或物体文件")
        return

    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # 创建增强管道
    color_aug_pipeline = create_color_light_augmentation()
    bg_shake_pipeline = create_shake_augmentation()

    all_images = []

    for i in range(num_images):
        # 选择背景并应用增强
        bg_path = os.path.join(backgrounds_dir, random.choice(bg_files))
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        # 先应用颜色增强
        bg_img = apply_augmentations(bg_img, color_aug_pipeline)
        # 再应用摇晃效果
        bg_img = apply_augmentations(bg_img, bg_shake_pipeline)

        bg_h, bg_w = bg_img.shape[:2]
        composite = bg_img.copy()
        occupied_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        annotations = []

        num_objs = random.randint(1, max_objects_per_image)

        for _ in range(num_objs):
            obj_file = random.choice(obj_files)
            obj_path = os.path.join(objects_dir, obj_file)
            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img is None:
                continue

            # 对物体应用颜色增强
            if obj_img.shape[2] == 4:
                rgb = obj_img[:, :, :3]
                alpha = obj_img[:, :, 3]
                augmented_rgb = apply_augmentations(rgb, color_aug_pipeline)
                obj_img = np.dstack((augmented_rgb, alpha))
            else:
                obj_img = apply_augmentations(obj_img, color_aug_pipeline)

            # 添加随机缩放
            scale = random.uniform(min_scale, max_scale)
            orig_h, orig_w = obj_img.shape[:2]
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # 确保缩放后的物体不会过大
            if new_w > bg_w * 0.8 or new_h > bg_h * 0.8:
                scale = min(bg_w * 0.8 / orig_w, bg_h * 0.8 / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

            if obj_img.shape[2] == 4:
                obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 添加随机旋转
            angle = random.uniform(-max_rotation, max_rotation)
            rotated_obj = rotate_image(obj_img, angle)
            rot_h, rot_w = rotated_obj.shape[:2]

            if rotated_obj.shape[2] == 4:
                obj_mask = (rotated_obj[:, :, 3] > 0).astype(np.uint8)
            else:
                obj_mask = np.ones((rot_h, rot_w), dtype=np.uint8)

            placed = False
            for _ in range(20):  # 尝试20次放置物体
                x = random.randint(0, bg_w - rot_w)
                y = random.randint(0, bg_h - rot_h)

                roi = occupied_mask[y:y + rot_h, x:x + rot_w]
                if np.sum(roi * obj_mask) == 0:
                    placed = True
                    break

            if not placed:
                continue

            occupied_mask[y:y + rot_h, x:x + rot_w] = np.maximum(
                occupied_mask[y:y + rot_h, x:x + rot_w],
                obj_mask
            )

            if rotated_obj.shape[2] == 4:
                obj_rgb = rotated_obj[:, :, :3]
                alpha = rotated_obj[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                composite[y:y + rot_h, x:x + rot_w] = \
                    (1 - alpha) * composite[y:y + rot_h, x:x + rot_w] + alpha * obj_rgb
            else:
                composite[y:y + rot_h, x:x + rot_w] = rotated_obj[:, :, :3]

            full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            full_mask[y:y + rot_h, x:x + rot_w] = obj_mask
            x_rect, y_rect, w_rect, h_rect = get_rotated_bbox(full_mask)

            x_center = (x_rect + w_rect / 2) / bg_w
            y_center = (y_rect + h_rect / 2) / bg_h
            width = w_rect / bg_w
            height = h_rect / bg_h

            class_id = get_class_id(obj_file)
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        img_name = f"image_{i:04d}.jpg"
        all_images.append(img_name)

        cv2.imwrite(os.path.join(output_dir, img_name), composite)

        with open(os.path.join(output_dir, img_name.replace(".jpg", ".txt")), 'w') as f:
            f.write("\n".join(annotations))

        print(f"生成: {img_name} (包含 {len(annotations)} 个物体)")

    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'elephant', 1: 'monkey', 2: 'peacock', 3: 'tiger', 4: 'wolf'},
        'nc': 5
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backgrounds_dir = os.path.join(script_dir, "backgrounds")
    objects_dir = os.path.join(script_dir, "objects")
    output_dir = os.path.join(script_dir, "yolo_dataset_final_final")

    create_yolo_dataset_with_rotation(
        backgrounds_dir=backgrounds_dir,
        objects_dir=objects_dir,
        output_dir=output_dir,
        num_images=500,
        max_objects_per_image=2,
        val_ratio=0.2,
        max_rotation=15,
        min_scale=0.5,  # 最小缩放比例
        max_scale=1.2  # 最大缩放比例
    )