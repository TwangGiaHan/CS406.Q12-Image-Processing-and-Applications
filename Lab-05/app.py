import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
import tempfile
import pandas as pd
from PIL import Image
import io

CLASS_NAMES = ['with_mask', 'without_mask']

# Màu sắc cho Bounding Box (BGR format cho OpenCV)
# Xanh lá cây cho with_mask (ID 0)
# Đỏ cho without_mask (ID 1)
BBOX_COLORS = {
    0: (0, 255, 0),  # Green (B, G, R)
    1: (0, 0, 255)   # Red (B, G, R)
}

@st.cache_resource # Cache mô hình để tránh tải lại mỗi lần cập nhật
def load_model(model_path):
    """Tải mô hình YOLOv11n đã huấn luyện."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()


MODEL_PATH = './best.pt'
model = load_model(MODEL_PATH)


def process_image(model, image_file, conf_thres, iou_thres):
    """Chạy dự đoán và vẽ Bounding Box lên ảnh."""
    
    # Đọc file ảnh dưới dạng mảng NumPy (OpenCV)
    image = Image.open(image_file).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Chạy mô hình dự đoán
    results = model.predict(
        source=img_cv, 
        conf=conf_thres, 
        iou=iou_thres, 
        verbose=False # Tắt output chi tiết của YOLO
    )
    
    # Lưu thống kê của ảnh này
    image_stats = {'with_mask': 0, 'without_mask': 0}
    
    # Xử lý kết quả dự đoán
    if results and results[0].boxes:
        boxes = results[0].boxes
        
        for box in boxes:
            # Lấy tọa độ BBox (x, y, w, h) và chuyển sang int
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            x1, y1, x2, y2 = xyxy
            label = CLASS_NAMES[class_id]
            color = BBOX_COLORS[class_id]
            
            # Cập nhật thống kê
            image_stats[label] += 1
            
            # Vẽ Bounding Box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Đặt nhãn (label và confidence)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(img_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Chuyển ảnh BGR của OpenCV về RGB để hiển thị trong Streamlit
    processed_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    return processed_img_rgb, image_stats

st.set_page_config(
    page_title="Mask Detection YOLOv8 Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("😷 Ứng dụng Phát hiện Tình trạng Đeo Khẩu trang")
st.markdown("Sử dụng mô hình **YOLOv8** đã huấn luyện để phân loại khuôn mặt thành: `with_mask` (Xanh lá) và `without_mask` (Đỏ).")
st.markdown("---")

# --- SIDEBAR (THAM SỐ TUỲ CHỈNH) ---

st.sidebar.header("⚙️ Tham số Mô hình")

# 1. Tùy chỉnh Confidence Threshold
conf_threshold = st.sidebar.slider(
    "Ngưỡng Tin cậy (Confidence Threshold)", 
    min_value=0.01, max_value=1.0, value=0.25, step=0.01,
    help="Giá trị càng cao, dự đoán càng chặt chẽ (ít dương tính giả)."
)

# 2. Tùy chỉnh IoU Threshold
iou_threshold = st.sidebar.slider(
    "Ngưỡng IoU (IoU Threshold)", 
    min_value=0.01, max_value=1.0, value=0.7, step=0.01,
    help="Giá trị càng cao, càng ít Bounding Box bị loại bỏ bởi Non-Maximum Suppression (NMS)."
)

st.sidebar.markdown("---")
st.sidebar.info("Mô hình đang chạy từ tệp: `best.pt`")

# --- UPLOAD VÀ XỬ LÝ ẢNH ---

uploaded_files = st.file_uploader(
    "🖼️ Tải lên một hoặc nhiều ảnh", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    
    # Khởi tạo tổng thống kê
    global_stats = {'with_mask': 0, 'without_mask': 0}
    
    # 1. Chạy xử lý ảnh
    results_list = []
    
    st.subheader(f"🔍 Kết quả Dự đoán ({len(uploaded_files)} ảnh)")
    
    # Chia giao diện thành 2 cột cho mỗi ảnh: Ảnh gốc và Ảnh đã xử lý
    cols = st.columns(2)

    for i, uploaded_file in enumerate(uploaded_files):
        
        # Xử lý ảnh và lấy thống kê
        processed_img, stats = process_image(model, uploaded_file, conf_threshold, iou_threshold)
        
        # Cập nhật tổng thống kê
        global_stats['with_mask'] += stats['with_mask']
        global_stats['without_mask'] += stats['without_mask']
        
        total_detected = stats['with_mask'] + stats['without_mask']
        
        # Lưu kết quả chi tiết cho Bảng thống kê
        results_list.append({
            'Tên Ảnh': uploaded_file.name,
            'Đeo đúng (0)': stats['with_mask'],
            'Sai cách/Không đeo (1)': stats['without_mask'],
            'Tổng': total_detected,
            '% Đúng': f"{stats['with_mask'] / total_detected * 100:.1f}%" if total_detected > 0 else '0.0%'
        })

        # --- Hiển thị kết quả từng ảnh ---
        st.markdown(f"#### Ảnh {i+1}: {uploaded_file.name}")
        
        st.image(processed_img, caption="Ảnh đã xử lý", use_container_width=True)
        
        # Hiển thị thống kê dưới dạng metrics
        st.markdown(f"##### Thống kê Ảnh {i+1}")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Tổng số người phát hiện", total_detected)
        col_m2.metric("Đeo khẩu trang đúng", stats['with_mask'])
        col_m3.metric("Đeo khẩu trang sai/Không đeo", stats['without_mask'])
        st.markdown("---")

    # --- 4. THỐNG KÊ TỔNG HỢP VÀ BIỂU ĐỒ ---
    
    st.header("📊 Thống kê Tổng hợp")
    
    total_batch = global_stats['with_mask'] + global_stats['without_mask']
    
    if total_batch > 0:
        pct_mask = global_stats['with_mask'] / total_batch
        pct_no_mask = global_stats['without_mask'] / total_batch
    else:
        pct_mask = 0
        pct_no_mask = 0

    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("Tổng số người trong Batch", total_batch)
    col_g2.metric("Tỉ lệ Đeo đúng", f"{pct_mask * 100:.1f}%", f"{global_stats['with_mask']} người")
    col_g3.metric("Tỉ lệ Sai/Không đeo", f"{pct_no_mask * 100:.1f}%", f"{global_stats['without_mask']} người")
    
    st.markdown("---")
    
    # 5. Bảng thống kê chi tiết
    st.subheader("Bảng thống kê Chi tiết")
    df_results = pd.DataFrame(results_list)
    st.dataframe(df_results)
    
    # 6. Biểu đồ tổng hợp
    st.subheader("Biểu đồ Phân bố Tổng thể")
    
    df_chart = pd.DataFrame({
        'Tình trạng': ['Đeo đúng (with_mask)', 'Sai/Không đeo (without_mask)'],
        'Số lượng': [global_stats['with_mask'], global_stats['without_mask']]
    })
    
    st.bar_chart(df_chart.set_index('Tình trạng'))
    
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu phân tích.")