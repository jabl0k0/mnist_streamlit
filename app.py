import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from model import create_and_load_model

st.set_page_config(page_title="MNIST Распознавание", layout="wide")
st.title("Распознавание рукописных цифр MNIST")

# Загрузка модели
@st.cache_resource
def load_model():
    return create_and_load_model()

model = load_model()

# ====================== Основной интерфейс ======================
col_left, col_right = st.columns([1.8, 1])

with col_left:
    st.subheader("Нарисуйте цифру")
    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=22,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

with col_right:
    st.subheader("Обработанное изображение (28×28)")
    processed_placeholder = st.empty()

# ====================== Предсказание ======================
if canvas_result.image_data is not None:
    # Преобразование изображения
    img = canvas_result.image_data
    pil_img = Image.fromarray(img.astype('uint8')).convert('L')
    pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
    pil_img = ImageOps.invert(pil_img)                    # инверсия цветов под MNIST
    
    # Показываем обработанное изображение справа
    processed_placeholder.image(pil_img, width=200)
    
    # Подготовка для модели
    img_array = np.array(pil_img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])   # (1, 28, 28, 1)
    
    # Предсказание
    prediction = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_digit]) * 100
    
    # Результаты
    st.success(f"**Предсказанная цифра: {predicted_digit}**")
    st.metric("Уверенность", f"{confidence:.1f}%")
    
    # График вероятностей
    st.bar_chart({str(i): float(p) for i, p in enumerate(prediction[0])})
