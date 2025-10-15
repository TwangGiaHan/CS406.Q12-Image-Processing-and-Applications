import os
import cv2
import pickle
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Image Search Engine", layout="wide")

# === Helpers ===
def process_image(image):
    """Trả về (hist_hsv, hist_rgb)"""
    # HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h0 = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    h1 = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
    h2 = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
    hist_hsv = np.concatenate((h0, h1, h2)).astype(np.float32)
    cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)

    # RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r0 = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    r1 = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    r2 = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    hist_rgb = np.concatenate((r0, r1, r2)).astype(np.float32)
    cv2.normalize(hist_rgb, hist_rgb, 0, 1, cv2.NORM_MINMAX)

    return hist_hsv, hist_rgb

@st.cache_data
def load_index(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)

def read_image_from_path(p):
    # Use np.fromfile to support Windows unicode paths
    try:
        arr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        return arr
    except Exception:
        return None

def compare_and_rank(query_hist, data_list, method):
    results = []
    for path, hist in data_list:
        try:
            score = float(cv2.compareHist(query_hist.astype(np.float32), hist.astype(np.float32), method))
            results.append((path, score))
        except Exception:
            continue
    # For some methods smaller is better
    small_is_better = method in (cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR, getattr(cv2, "HISTCMP_CHISQR_ALT", None))
    results.sort(key=lambda x: x[1], reverse=not small_is_better)
    return results

# === UI ===
st.title("Image Search Engine — Histogram-based")

base_dir = Path.cwd()

@st.cache_resource
def load_database():
    with open("./data_hsv.pkl", "rb") as f:
        db_hsv = pickle.load(f)
    with open("./data_rgb.pkl", "rb") as f:
        db_rgb = pickle.load(f)
    return db_hsv, db_rgb

loaded_data_hsv, loaded_data_rgb = load_database()

st.sidebar.markdown("### Cấu hình tìm kiếm")
color_space = st.sidebar.selectbox("Không gian màu:", ["HSV", "RGB"])
method_name = st.sidebar.selectbox("Phương pháp so sánh:", ["CORREL", "CHISQR", "BHATTACHARYYA", "INTERSECT"])
top_k = st.sidebar.slider("Top K results", 1, 12, 6)

# Map method name to cv2 constant
method_map = {
    "CORREL": cv2.HISTCMP_CORREL,
    "CHISQR": cv2.HISTCMP_CHISQR,
    "BHATTACHARYYA": cv2.HISTCMP_BHATTACHARYYA,
    "INTERSECT": cv2.HISTCMP_INTERSECT
}
method = method_map.get(method_name, cv2.HISTCMP_CORREL)

st.sidebar.markdown("---")
st.sidebar.write(f"HSV index: {len(loaded_data_hsv)} items")
st.sidebar.write(f"RGB index: {len(loaded_data_rgb)} items")

uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])

# Optional: let user pick a sample from loaded index as query
sample_query = None
if (color_space == "HSV" and loaded_data_hsv) or (color_space == "RGB" and loaded_data_rgb):
    st.markdown("Chọn ảnh từ dataset (tuỳ chọn):")
    sample_list = (loaded_data_hsv if color_space=="HSV" else loaded_data_rgb)[:20]
    cols = st.columns(5)
    for i, (p, _) in enumerate(sample_list):
        try:
            img = read_image_from_path(p)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with cols[i%5]:
                if st.button(Path(p).name, key=p):
                    sample_query = p
                st.image(img_rgb, use_container_width=True)
        except Exception:
            continue

query_img = None
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif sample_query:
    query_img = read_image_from_path(sample_query)

if query_img is not None:
    st.subheader("Query image")
    st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), use_container_width=False, width=300)

    hist_hsv, hist_rgb = process_image(query_img)
    if color_space == "HSV":
        data_list = loaded_data_hsv
        qhist = hist_hsv
    else:
        data_list = loaded_data_rgb
        qhist = hist_rgb

    if not data_list:
        st.warning("Không có index cho không gian màu đã chọn. Kiểm tra đường dẫn pickle.")
    else:
        results = compare_and_rank(qhist, data_list, method)[:top_k]
        st.subheader(f"Top-{top_k} similar images ({color_space}, {method_name})")
        cols = st.columns(min(top_k, 6))
        for i, (path, score) in enumerate(results):
            try:
                # đường dẫn tương đối từ thư mục dataset
                path = os.path.join("dataset", path.lstrip("./"))
                img = read_image_from_path(path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with cols[i % len(cols)]:
                    st.image(img_rgb, caption=f"{Path(path).name}\nScore: {score:.4f}", use_container_width=True)
            except Exception:
                continue
else:
    st.info("Upload ảnh truy vấn hoặc chọn 1 ảnh từ dataset để bắt đầu truy vấn.")