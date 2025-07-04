import streamlit as st
import cv2
import numpy as np
import io
import zipfile
import random

st.set_page_config(page_title="✨ Image Augmenter", layout="centered")

st.markdown("""
    <style>
    stApp {
        background-color: #f0f4f8;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        max-width: 800px;
        margin: auto;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #333333;
        margin-bottom: 2rem;
        animation: fadeInUp 1.5s ease-out;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #004999;
        transform: scale(1.05);
    }
    @keyframes fadeInDown {
        0% { transform: translateY(-20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeInUp {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.header("🔧 Augmentation Settings")
transform_options = st.sidebar.multiselect(
    "🔄 Choose transformations (can select multiple)",
    ["Translation", "Cropping", "Shearing", "Rotation", "Scaling",
     "Grayscale", "Flip Horizontally", "Flip Vertically"],
    placeholder="Enter multiple transformations"
)
count = st.sidebar.slider("📈 Number of images to generate", 10, 200, 100, step=10)


st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📸 Image Augmentation App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image, choose transformations, preview, and download!</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

def apply_multiple_transformations(image, options, count):
    rows, cols = image.shape[:2]
    transformed_images = []

    for _ in range(count):
        img = image.copy()
        for option in options:
            if option == "Translation":
                tx, ty = random.randint(-50, 50), random.randint(-50, 50)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img, M, (cols, rows))

            elif option == "Cropping":
                x1 = random.randint(0, cols // 4)
                y1 = random.randint(0, rows // 4)
                x2 = random.randint(3 * cols // 4, cols)
                y2 = random.randint(3 * rows // 4, rows)
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (cols, rows))

            elif option == "Shearing":
                shear_factor = random.uniform(-0.3, 0.3)
                M = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
                img = cv2.warpAffine(img, M, (cols, rows))

            elif option == "Rotation":
                angle = random.randint(-45, 45)
                center = (cols // 2, rows // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (cols, rows))

            elif option == "Scaling":
                scale = random.uniform(0.5, 1.5)
                resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                img = cv2.resize(resized, (cols, rows))

            elif option == "Grayscale":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif option == "Flip Horizontally":
                img = cv2.flip(img, 1)

            elif option == "Flip Vertically":
                img = cv2.flip(img, 0)

        transformed_images.append(img)

    return transformed_images

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("❌ Failed to decode the image. Please upload a valid image file.")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="📷 Original Image", use_container_width=True)

    if st.button("✨ Generate Augmented Images"):
        with st.spinner("Generating images..."):
            augmented_images = apply_multiple_transformations(image, transform_options, count)

            st.markdown("### 🔍 Preview of Augmented Images")
            preview_cols = st.columns(3)
            for i in range(min(3, len(augmented_images))):
                preview_img = cv2.cvtColor(augmented_images[i], cv2.COLOR_BGR2RGB)
                preview_cols[i].image(preview_img, use_container_width=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, img in enumerate(augmented_images):
                    is_success, buffer = cv2.imencode(".png", img)
                    zip_file.writestr(f"augmented_{idx+1}.png", buffer.tobytes())
            zip_buffer.seek(0)

        st.success(f"✅ {count} Augmented Images Generated!")
        st.download_button("📁 Download ZIP", zip_buffer, "augmented_images.zip", "application/zip")

st.markdown('</div>', unsafe_allow_html=True)
