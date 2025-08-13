import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
# Load model (ví dụ: model Keras đã train trên MNIST)
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
        img = canvas_result.image_data

        # Chuyển sang PIL để resize
        img = Image.fromarray((img[:, :, 0:3]).astype(np.uint8))  # Bỏ alpha
        img = img.convert("L")  # Chuyển grayscale
        img = img.resize((28, 28), Image.LANCZOS)

        # Chuẩn hóa dữ liệu cho model
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape((1, 28 * 28))

        # Suy luận
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.write(f"**Kết quả dự đoán:** {predicted_digit}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Vui lòng vẽ một chữ số trước!")
