import sys
import os
import argparse
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO  # 确保该路径指向正确的模块

def resize_and_letterbox(image, target_size=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to maintain aspect ratio."""
    img = image.resize((target_size[1], target_size[0]), Image.BILINEAR)  # Resize using PIL
    img = np.array(img)  # Convert to numpy array for further processing
    shape = img.shape[:2]  # current shape [height, width]
    dw, dh = target_size[1] - shape[1], target_size[0] - shape[0]  # Calculate padding
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_CONSTANT, value=color)  # Add padding
    return img

def detect(image_path, model_path="C:\\Users\\18301\\OneDrive\\Desktop\\FractureDetection1.2\\fracture_detection\\yolov8m_10.16.pt", conf_threshold=0.5):
    """Load the model and perform detection on the given image."""
    model = YOLO(model_path)

    # 预处理图片
    try:
        original_img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return [], None

    img = resize_and_letterbox(original_img)

    # 模型推理
    results = model(img)

    # 提取检测结果
    boxes = []
    original_width, original_height = original_img.size  # 原图的宽高

    for result in results:  # 遍历每个检测结果
        if result.boxes is not None:
            for box in result.boxes.data.tolist():  # 获取检测框数据
                x1, y1, x2, y2, conf, cls = box
                # 限制坐标在图像边界内
                x1 = min(max(int(x1), 0), original_width - 1)
                x2 = min(max(int(x2), 0), original_width - 1)
                y1 = min(max(int(y1), 0), original_height - 1)
                y2 = min(max(int(y2), 0), original_height - 1)

                if conf >= conf_threshold:  # 只添加高于置信度阈值的框
                    boxes.append([x1, y1, x2, y2, conf, cls])  # 存储为整数类型的坐标和置信度

    return boxes, original_img

def main():
    parser = argparse.ArgumentParser(description='Fracture Detection with YOLOv8')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    boxes, original_img = detect(args.image_path)
    if original_img is not None:
        for det in boxes:
            x1, y1, x2, y2, conf, cls = det
            print(f'Detected: Class {cls} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]')

if __name__ == "__main__":
    main()
