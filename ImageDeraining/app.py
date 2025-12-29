import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from streamlit_image_comparison import image_comparison
import os

from model import UnetGenerator 


st.set_page_config(
    page_title="Single Image Deraining - Pix2Pix",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./models/generator_best.pth"   

@st.cache_resource
def load_model():
    model = UnetGenerator().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

generator = load_model()


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [-1,1]
])

def pad_to_multiple(img, multiple=256):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img_padded, h, w

def run_inference(image_pil):
    img = np.array(image_pil).astype(np.float32) / 255.0
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

    tensor_pad, h, w = pad_to_multiple(tensor)

    with torch.no_grad():
        fake_pad = generator(tensor_pad)

    fake = fake_pad[:, :, :h, :w]

    fake = torch.clamp((fake + 1) / 2, 0, 1)

    fake_img = fake.squeeze(0).cpu().permute(1, 2, 0).numpy()
    fake_img = (fake_img * 255).astype(np.uint8)

    return fake_img


# UI
st.title("🌧️ Single Image Deraining Demo")
st.markdown(
    """
    **Mô hình:** Pix2Pix (GAN-based)  
    **Chức năng:** Xóa mưa từ một ảnh đầu vào  
    **So sánh:** Kéo thanh trượt để so sánh Input và Output
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload ảnh có mưa",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("⏳ Đang xử lý ảnh..."):
        output_image = run_inference(input_image)

    st.subheader("🔍 So sánh kết quả")

    # IMAGE SLIDER (BEFORE / AFTER)
    image_comparison(
        img1=input_image,
        img2=output_image,
        label1="Rainy Input",
        label2="Derained Output",
        width=800,
        starting_position=50
    )

    st.download_button(
        label="⬇️ Tải ảnh đã xóa mưa",
        data=Image.fromarray(output_image).tobytes(),
        file_name="derained.png",
        mime="image/png"
    )

else:
    st.info("👆 Vui lòng upload một ảnh có mưa để bắt đầu.")

st.markdown("---")
st.markdown(
    "📌 **Deraining Demo** — Built with Streamlit & PyTorch"
)
