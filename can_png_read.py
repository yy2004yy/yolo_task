import cv2
import os
def check_png_files(objects_dir):
    """检查PNG文件是否可读"""
    for file in os.listdir(objects_dir):
        if file.lower().endswith('.png'):
            file_path = os.path.join(objects_dir, file)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"❌ 无法读取: {file}")
            else:
                print(f"✅ 可读取: {file} - 尺寸: {img.shape}")

# 使用示例
objects_dir = os.path.join(os.path.dirname(__file__), "objects")
check_png_files(objects_dir)