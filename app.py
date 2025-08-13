import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import matplotlib.pyplot as plt
# Load model (ví dụ: model Keras đã train trên MNIST)

def preprocess_canvas_image(img_pil):
    # 1. Chuyển sang grayscale
    img = img_pil.convert("L")

    # 2. Invert nếu cần (trắng nét, đen nền)
    # img = ImageOps.invert(img)

    # 3. Lấy bounding box của nét vẽ
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 4. Resize nét vẽ về 20x20
    img = img.resize((20, 20), Image.LANCZOS)

    # 5. Tạo nền đen 28x28 và dán ảnh vào giữa
    background = Image.new("L", (28, 28), (0))
    background.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

    # 6. Chuẩn hóa về [0,1]
    img_array = np.array(background) / 255.0
    img_array = img_array.reshape(1, 28 * 28)

    return img_array


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

st.title("🖌️ MNIST Digit Recognizer Demo")
st.write("Vẽ một chữ số (0 - 9) vào bảng bên dưới rồi nhấn **Dự đoán**")

# Kích thước canvas
CANVAS_SIZE = 280

# Bảng vẽ
canvas_result = st_canvas(
    fill_color="#000000",  # Màu tô (đen)
    stroke_width=15,       # Độ dày nét
    stroke_color="#FFFFFF",# Màu nét (trắng)
    background_color="#000000", # Nền đen
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Dự đoán"):
    if canvas_result.image_data is not None:
        # Lấy ảnh từ canvas
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))
        img_array = preprocess_canvas_image(img)
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.write(f"**Kết quả dự đoán:** {predicted_digit}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Vui lòng vẽ một chữ số trước!")
