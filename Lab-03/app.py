import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Image Enhancing")

# --- Functions ---
def add_gaussian_noise(img):
    row, col, ch = img.shape
    mean = 0
    sigma = 20
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def denoise_mean(img):
    return cv2.blur(img, (5, 5))

def denoise_median(img):
    return cv2.medianBlur(img, 5)

def denoise_color(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def Sharpening(img):
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edge_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_img

def prewitt_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])
    grad_x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    grad_y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edge_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_img

def candy_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

# --- UI ---
st.title("Image Enhancing — Denoise · Sharpen · Edge Detection")

uploaded = st.file_uploader("Upload a color image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image[:, :, ::-1], use_container_width=True)

    # --- Add Noise ---
    noisy_image = add_gaussian_noise(image)
    st.subheader("Noisy Image (Gaussian Noise)")
    st.image(noisy_image[:, :, ::-1], use_container_width=True)

    # --- Denoising Methods ---
    st.subheader("Denoised Images")
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        mean_denoised = denoise_mean(noisy_image)
        st.image(mean_denoised[:, :, ::-1], caption="Mean Filter (Average Blur)", use_container_width=True)
    
    with col_d2:
        median_denoised = denoise_median(noisy_image)
        st.image(median_denoised[:, :, ::-1], caption="Median Filter", use_container_width=True)
    
    with col_d3:
        color_denoised = denoise_color(noisy_image)
        st.image(color_denoised[:, :, ::-1], caption="FastNlMeans Denoising", use_container_width=True)

    # --- Sharpen ---
    st.subheader("Sharpened Image")
    sharpened_image = Sharpening(image)
    st.image(sharpened_image[:, :, ::-1], use_container_width=True)

    # --- Edge Detection ---
    st.subheader("Edge Detection Outputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        sobel_image = sobel_edge_detection(image)
        st.image(sobel_image, caption="Sobel Edge Detection", use_container_width=True)

    with col2:
        prewitt_image = prewitt_edge_detection(image)
        st.image(prewitt_image, caption="Prewitt Edge Detection", use_container_width=True)

    with col3:
        candy_image = candy_edge_detection(image)
        st.image(candy_image, caption="Canny Edge Detection", use_container_width=True)
