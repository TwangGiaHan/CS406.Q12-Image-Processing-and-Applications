import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import time

# --- Cấu hình Cố định ---
IMG_SIZE = (224, 224)
MODEL_PATHS = {
    "VGG16": "models/VGG16_Intel.h5",
    "ResNet50": "models/ResNet50_Intel.h5",
    "EfficientNetB0": "models/EfficientNetB0_Intel.h5"
}
CLASS_LABELS = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
PREPROCESS_FUNCS = {
    'VGG16': vgg_preprocess,
    'ResNet50': resnet_preprocess,
    'EfficientNetB0': eff_preprocess,
}

# --- Hàm Tải Mô hình (Sử dụng caching) ---
@st.cache_resource
def load_all_models():
    """Tải tất cả các mô hình và lưu vào cache."""
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            model = load_model(path)
            models[name] = model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình {name} từ {path}: {e}")
            st.stop()
    return models

# --- Hàm Tiền xử lý và Dự đoán ---
def predict_image(model, preprocess_func, img_path):
    """Tiền xử lý ảnh, thực hiện dự đoán và đo thời gian suy luận."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    processed_img = preprocess_func(img_array)

    start_time = time.perf_counter()
    predictions = model.predict(processed_img, verbose=0)
    end_time = time.perf_counter()
    
    inference_time = end_time - start_time

    scores = predictions[0] * 100
    
    max_index = np.argmax(scores)
    prediction = CLASS_LABELS[max_index]
    confidence = scores[max_index]

    return prediction, confidence, scores, inference_time

# --- Main Streamlit App ---

st.set_page_config(
    page_title="So Sánh & Triển Khai Mô Hình Phân Loại Ảnh",
    layout="wide"
)

st.title("🖼️ Triển Khai và So Sánh Mô Hình Phân Loại Ảnh")
st.markdown("Sử dụng **VGG16, ResNet50, EfficientNetB0** để phân loại 6 loại ảnh vệ tinh (Intel Image Classification). So sánh về hiệu suất và Thời gian Suy luận (Inference Time)")

# --- Tải Mô hình ---
models = load_all_models()
if not models:
    st.error("Không có mô hình nào được tải thành công. Vui lòng kiểm tra lại đường dẫn file .h5.")
    st.stop()

st.divider()

st.header("📸 Triển khai Mô hình Demo")
st.sidebar.header("Tải Ảnh Lên")

uploaded_file = st.sidebar.file_uploader(
    "Chọn một ảnh...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.subheader("Ảnh Đã Tải Lên")
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption='Ảnh đầu vào.', use_container_width=True)

    image_file = io.BytesIO(image_bytes)

    st.subheader("Kết quả Dự đoán")
    
    cols = st.columns(3)
    
    results = {}
    
    for i, (name, model) in enumerate(models.items()):
        
        preprocess_func = PREPROCESS_FUNCS[name]
        
        prediction, confidence, scores, inference_time = predict_image(model, preprocess_func, image_file)
        
        results[name] = (prediction, confidence, scores, inference_time)

        with cols[i]:
            st.info(f"**{name}**")            
            st.metric(label="Phân Loại", value=prediction, delta=f"{confidence:.2f}%")
            st.metric(label="Thời gian Suy luận (s)", value=f"{inference_time:.3f} s")
            # Vẽ biểu đồ phân phối xác suất
            fig, ax = plt.subplots()
            y_pos = np.arange(len(CLASS_LABELS))
            ax.barh(y_pos, scores, align='center', color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_LABELS)
            ax.invert_yaxis()
            ax.set_xlabel('Xác suất (%)')
            ax.set_title(f'Xác suất Lớp - {name}')
            
            st.pyplot(fig)
            
else:
    st.info("Vui lòng tải một tệp ảnh để xem kết quả dự đoán của các mô hình.")
    
st.divider()
st.caption("Ứng dụng demo so sánh các kiến trúc CNN: VGG16, ResNet50, EfficientNetB0. Thực hiện bởi **Tăng Gia Hân**.")