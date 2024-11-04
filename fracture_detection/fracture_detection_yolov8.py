import os
import zipfile
from io import BytesIO
import shutil
import concurrent.futures
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# 定义上传路径和输出路径
UPLOAD_FOLDER = "uploads/"
OUTPUT_FOLDER = "output/"
MAX_SIZE = 5 * 1024 * 1024  # 设置最大上传文件大小为 5MB

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载模型并进行推理
def detect(image_path, model, conf_threshold=0.4):
    results = model(image_path)
    boxes = []
    original_img = Image.open(image_path)

    if results:
        # 遍历检测结果中的每个 box
        for result in results:
            for box in result.boxes:
                if box.conf[0] > conf_threshold:  # 仅考虑置信度高于阈值的框
                    boxes.append(box.xyxy[0].int().tolist() + [box.conf[0].item(), box.cls[0].item()])  # 直接提取信息

    return boxes, original_img

# 结果可视化
def visualize_detection(original_img, boxes):
    img = np.array(original_img)
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"Class: {cls}, Conf: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# 保存上传的图片
def save_uploadedfile(uploadedfile):
    file_path = os.path.join(UPLOAD_FOLDER, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

# 压缩检测结果为 zip 文件
def zip_output(output_folder):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for foldername, _, filenames in os.walk(output_folder):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                zf.write(filepath, arcname=os.path.relpath(filepath, output_folder))
    zip_buffer.seek(0)
    return zip_buffer

# 处理单张图片的函数
def process_image(uploaded_file, model):
    image_path = save_uploadedfile(uploaded_file)
    boxes, original_img = detect(image_path, model)
    result_img = visualize_detection(original_img, boxes)
    result_img_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name)
    Image.fromarray(result_img).save(result_img_path)
    return uploaded_file.name, result_img

# Streamlit 界面
def main():
    st.title("Fracture Detection")
    uploaded_files = st.file_uploader("Select photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        photo_word = "photo" if len(uploaded_files) == 1 else "photos"
        st.write(f"{len(uploaded_files)} {photo_word} uploaded")

        # 检查文件大小
        for uploaded_file in uploaded_files:
            if uploaded_file.size > MAX_SIZE:
                st.error(f"{uploaded_file.name} is too large! Max size is {MAX_SIZE / (1024 * 1024):.2f} MB.")
                return

        if st.button("Start detection"):
            st.write("Fracture detection in progress...")
            progress = st.progress(0)
            progress_text = st.empty()  # 用于显示百分比

            model = YOLO('fracture_detection/yolov8m_10.16.pt')  # Load model here

            # 使用线程池并行处理每张图片
            images_to_display = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, uploaded_file, model): uploaded_file for uploaded_file in uploaded_files}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    images_to_display.append(future.result())
                    progress.progress((i + 1) / len(uploaded_files))  # 更新进度条
                    progress_text.text(f"Progress: {((i + 1) / len(uploaded_files)) * 100:.1f}%")  # 更新百分比

            num_images = len(images_to_display)
            num_columns = 5  # 每行显示5张图片
            rows = (num_images + num_columns - 1) // num_columns

            for row in range(rows):
                cols = st.columns(num_columns)
                for col in range(num_columns):
                    index = row * num_columns + col
                    if index < num_images:
                        file_name, result_img = images_to_display[index]
                        with cols[col]:
                            st.image(result_img, use_column_width='always')
                            st.write(f"{file_name} Test results")  # 文字说明在下方
                    else:
                        with cols[col]:  # 处理空白列
                            st.write("")

            st.success("Testing completed!")
            zip_file = zip_output(OUTPUT_FOLDER)
            st.download_button(label="Download test results", data=zip_file, file_name="detection_results.zip", mime="application/zip")

            # 清理上传和输出文件夹（可选）
            if st.button("Clean up files"):
                shutil.rmtree(UPLOAD_FOLDER)
                shutil.rmtree(OUTPUT_FOLDER)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                st.success("Files cleaned")

if __name__ == "__main__":
    main()
