# Lab-05 — MASK WEARING DETECTION

## Nội dung thư mục

- eda-maskfacedataset.ipynb — các Notebook phân tích, tiền xử lý, training hoặc EDA (ví dụ: eda-maskfacedataset.ipynb)
- data/ — tải tại link: https://drive.google.com/file/d/1nng7lMDXqTT4Hrmje0BCSf5nq-9jc3QF/view?usp=sharing

## Yêu cầu

- Python 3.8+
- Khuyến nghị tạo virtual environment
- Các thư viện cơ bản:
  - numpy, pandas, matplotlib, seaborn
  - opencv-python, pillow
  - scikit-learn
  - tensorflow / torch
  - ultralytics
  - lxml

## Thiết lập nhanh (Windows)

1. Mở terminal tại thư mục gốc project:
   cd d:\CS406

2. Tạo và kích hoạt venv:
   python -m venv .venv
   .venv\Scripts\activate

3. Cài đặt packages:
   pip install -r requirements.txt
   hoặc
   pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn

## Chạy Notebook / Script

- Mở VS Code hoặc Jupyter Notebook và load notebook `eda-maskfacedataset.ipynb`.
- Đảm bảo kernel/interpreter trỏ tới virtualenv đã cài package.
- Nếu notebook có cell dùng đường dẫn tới dataset, chỉnh biến PATH tương ứng.

## Các bước thực hiện

- EDA: kiểm tra số lượng ảnh, phân bố bounding box, phân bố nhãn.
- Chuyển đổi annotation (Pascal VOC XML → YOLO txt).
- Chia train/val/test và sao chép ảnh + nhãn vào cấu trúc YOLO.
- (Tùy chọn) Train model (Ultralytics YOLO) — xem cell hướng dẫn training trong notebook.

## Kiểm tra & Debug

- Nếu notebook báo "file not found": kiểm tra biến đường dẫn (IMAGE_DIR, ANNOTATIONS_DIR).
- Khi chuyển XML → YOLO, kiểm tra:
  - tọa độ bbox hợp lệ (xmin < xmax, ymin < ymax)
  - ảnh tồn tại với cùng tên (png/jpg)
- Kiểm tra phân bố lớp sau khi gộp (nếu có) để đảm bảo stratified split hoạt động.

## Đặt file mô hình

Đặt file best.pt (YOLOv11n đã train xong) vào thư mục gốc cùng app.py

## Chạy ứng dụng

Chạy câu lệnh:
`streamlit run app.py`

Hệ thống sẽ mở giao diện tại địa chỉ:
`http://localhost:8501`
