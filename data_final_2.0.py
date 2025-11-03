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
    """创建专注于颜色和光照变化的增强管道"""
    return A.Compose([
        # ==================== 颜色变换 ====================
        A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=40,
            val_shift_limit=20,
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.2,
            p=0.7
        ),
        A.CLAHE(
            clip_limit=3.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),
        A.Solarize(
            threshold=200,
            p=0.1
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            shadow_dimension=3,
            p=0.2
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            num_flare_circles_lower=1,
            num_flare_circles_upper=6,
            src_radius=50,
            src_color=(200, 200, 200),
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=(0.1, 2.0),
            p=0.25
        ),
        A.Downscale(
            scale_min=0.5,
            scale_max=0.9,
            interpolation=cv2.INTER_LINEAR,
            p=0.2
        ),
    ])


def create_shake_augmentation():
    """创建背景摇晃(相机抖动)增强管道"""
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.0,
            rotate_limit=2,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        )
    ])


def add_stripes_pattern(image, intensity=0.3):
    """添加随机条纹图案"""
    h, w = image.shape[:2]
    stripes = np.zeros((h, w, 3), dtype=np.uint8)

    # 随机选择条纹方向 (0-垂直, 1-水平, 2-对角)
    direction = random.randint(0, 2)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if direction == 0:  # 垂直条纹
        stripe_width = random.randint(5, 30)
        for x in range(0, w, stripe_width):
            if random.random() > 0.5:
                stripes[:, x:x + stripe_width] = color
    elif direction == 1:  # 水平条纹
        stripe_height = random.randint(5, 30)
        for y in range(0, h, stripe_height):
            if random.random() > 0.5:
                stripes[y:y + stripe_height, :] = color
    else:  # 对角条纹
        stripe_width = random.randint(10, 40)
        for d in range(-h, w, stripe_width):
            cv2.line(stripes, (d, 0), (d + h, h), color, stripe_width)

    return cv2.addWeighted(image, 1 - intensity, stripes, intensity, 0)


def add_gradient_overlay(image, intensity=0.3):
    """添加渐变叠加效果"""
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # 随机选择渐变类型 (0-线性, 1-径向)
    grad_type = random.randint(0, 1)
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if grad_type == 0:  # 线性渐变
        direction = random.randint(0, 3)
        if direction == 0:  # 从左到右
            for x in range(w):
                alpha = x / w
                overlay[:, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
        elif direction == 1:  # 从上到下
            for y in range(h):
                alpha = y / h
                overlay[y, :] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
        else:  # 对角线
            for y in range(h):
                for x in range(w):
                    alpha = (x + y) / (w + h)
                    overlay[y, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
    else:  # 径向渐变
        center_x, center_y = random.randint(0, w), random.randint(0, h)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                alpha = dist / max_dist
                overlay[y, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)

    return cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)


def add_light_spot(image, intensity=0.5):
    """添加强光斑效果"""
    h, w = image.shape[:2]
    light = np.zeros((h, w, 3), dtype=np.float32)

    # 随机光斑位置
    center_x = random.randint(0, w)
    center_y = random.randint(0, h)
    radius = random.randint(min(h, w) // 4, min(h, w) // 2)
    color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist < radius:
                factor = 1 - (dist / radius) ** 2
                light[y, x] = factor * np.array(color)

    light = light.astype(np.uint8)
    return cv2.addWeighted(image, 1 - intensity, light, intensity, 0)


def create_random_color_background(h, w):
    """创建随机颜色背景"""
    # 随机选择背景类型 (0-纯色, 1-渐变, 2-噪点)
    bg_type = random.randint(0, 2)

    if bg_type == 0:  # 纯色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        bg = np.full((h, w, 3), color, dtype=np.uint8)
    elif bg_type == 1:  # 渐变
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])

        if direction == 'horizontal':
            for x in range(w):
                alpha = x / w
                bg[:, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
        elif direction == 'vertical':
            for y in range(h):
                alpha = y / h
                bg[y, :] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
        elif direction == 'diagonal':
            for y in range(h):
                for x in range(w):
                    alpha = (x + y) / (w + h)
                    bg[y, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
        else:  # radial
            center_x, center_y = w // 2, h // 2
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    alpha = dist / max_dist
                    bg[y, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
    else:  # 噪点
        bg = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    return bg


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


def apply_special_effects(image, prob=0.5):
    """应用特殊效果(条纹、渐变、光斑)"""
    if random.random() < prob:
        effect_type = random.choice(['stripes', 'gradient', 'light_spot', 'combo'])

        if effect_type == 'stripes':
            image = add_stripes_pattern(image, intensity=random.uniform(0.1, 0.4))
        elif effect_type == 'gradient':
            image = add_gradient_overlay(image, intensity=random.uniform(0.1, 0.3))
        elif effect_type == 'light_spot':
            image = add_light_spot(image, intensity=random.uniform(0.2, 0.5))
        else:  # combo
            if random.random() > 0.5:
                image = add_stripes_pattern(image, intensity=random.uniform(0.1, 0.3))
            if random.random() > 0.5:
                image = add_gradient_overlay(image, intensity=random.uniform(0.1, 0.2))
            if random.random() > 0.5:
                image = add_light_spot(image, intensity=random.uniform(0.1, 0.3))

    return image


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
    """创建带旋转增强、颜色/光照噪声和背景摇晃的标准YOLO数据集"""
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
        # 随机决定是否使用纯色背景
        use_color_bg = random.random() < 0.3  # 30%概率使用纯色背景

        if not use_color_bg and bg_files:
            # 使用真实背景图片
            bg_path = os.path.join(backgrounds_dir, random.choice(bg_files))
            bg_img = cv2.imread(bg_path)
            if bg_img is None:
                continue
            bg_h, bg_w = bg_img.shape[:2]
        else:
            # 创建随机颜色背景
            bg_h, bg_w = random.randint(640, 1920), random.randint(640, 1920)
            bg_img = create_random_color_background(bg_h, bg_w)

        # 先应用颜色增强
        bg_img = apply_augmentations(bg_img, color_aug_pipeline)
        # 再应用摇晃效果
        bg_img = apply_augmentations(bg_img, bg_shake_pipeline)
        # 应用特殊效果
        bg_img = apply_special_effects(bg_img, prob=0.4)

        composite = bg_img.copy()
        occupied_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        annotations = []

        num_objs = random.randint(1, max_objects_per_image)
        obj_count = 0

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

            # 对物体应用特殊效果
            if obj_img.shape[2] == 4:
                rgb = obj_img[:, :, :3]
                alpha = obj_img[:, :, 3]
                rgb = apply_special_effects(rgb, prob=0.3)
                obj_img = np.dstack((rgb, alpha))
            else:
                obj_img = apply_special_effects(obj_img, prob=0.3)

            # 添加随机缩放
            orig_h, orig_w = obj_img.shape[:2]

            # 计算最大允许缩放比例
            max_allowed_scale = min(
                (bg_w * 0.8) / orig_w,
                (bg_h * 0.8) / orig_h
            )

            # 确保缩放比例在合理范围内
            scale = random.uniform(min_scale, min(max_scale, max_allowed_scale))
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

            # 检查旋转后的物体是否仍然适合背景
            if rot_w > bg_w or rot_h > bg_h:
                # 如果旋转后物体太大，尝试调整缩放比例
                scale = min(bg_w / rot_w, bg_h / rot_h) * 0.9  # 再缩小10%确保有空间
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                if obj_img.shape[2] == 4:
                    obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                rotated_obj = rotate_image(obj_img, angle)
                rot_h, rot_w = rotated_obj.shape[:2]

                # 如果仍然太大，跳过这个物体
                if rot_w > bg_w or rot_h > bg_h:
                    continue

            if rotated_obj.shape[2] == 4:
                obj_mask = (rotated_obj[:, :, 3] > 0).astype(np.uint8)
            else:
                obj_mask = np.ones((rot_h, rot_w), dtype=np.uint8)

            placed = False
            for _ in range(20):  # 尝试20次放置物体
                try:
                    x = random.randint(0, max(0, bg_w - rot_w))
                    y = random.randint(0, max(0, bg_h - rot_h))
                except ValueError:
                    continue  # 如果仍然有问题，跳过这次尝试

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
            obj_count += 1

        if obj_count == 0:
            continue  # 如果没有成功放置任何物体，跳过这张图片

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
    output_dir = os.path.join(script_dir, "yolo_dataset_final_final_final_v6.0")

    create_yolo_dataset_with_rotation(
        backgrounds_dir=backgrounds_dir,
        objects_dir=objects_dir,
        output_dir=output_dir,
        num_images=500,
        max_objects_per_image=2,
        val_ratio=0.2,
        max_rotation=160,
        min_scale=0.5,
        max_scale=1.2
    )