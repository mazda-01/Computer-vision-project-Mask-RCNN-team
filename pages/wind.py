import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import time
import os

st.markdown("""
     <style>
    /* Скрываем заголовок "Pages" */
    [data-testid="stSidebar"] > div:first-child > div:first-child > h2 {
        display: none;
    }
    
    /* Скрываем список страниц */
    [data-testid="stSidebar"] > div:first-child > div:nth-child(2) {
        display: none;
    }
    
    /* Если нужно — скрываем разделитель */
    [data-testid="stSidebar"] > div:first-child > hr {
        display: none;
    }        
    
    /* Основной контейнер */
    .block-container {
        max-width: 1300px !important;
        padding: 2rem 2rem !important;
        margin: 0 auto;
    }

    /* Фон страницы */
    .stApp {
        background-image: url("https://image.fonwall.ru/o/zp/sky-road-street-windmill.jpeg?auto=compress&fit=resize&w=1200&h=806&display=large&domain=img3.fonwall.ru");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    /* Тёмные карточки */
    .css-1v0mbdj, .css-12w0y3b, .stMarkdown, .stTabs, .stDataFrame, 
    .stPlotlyChart, .stImage, .stTable, div[data-testid="stHorizontalBlock"] {
        background-color: rgba(44, 91, 94, 0.88) !important;
        color: #f1f5f9 !important;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Текст */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, 
    .stMarkdown p, .stMarkdown li {
        color: #f1f5f9 !important;
    }

    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(44, 91, 94, 0.6);
        border-radius: 10px 10px 0 0;
        color: #cbd5e1;
        font-weight: 600;
        padding: 0 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #438e94;
        color: #e2e8f0;
    }

    .stPlotlyChart, .stPyplot {
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# ЗАГРУЗКА МОДЕЛИ
# ----------------------------
@st.cache_resource
def load_model():
    model_path = "models/wind.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Модель не найдена: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

model = load_model()

# Настройка страницы
st.set_page_config(page_title="Детекция ветряных турбин", layout="centered")

# Вкладки
tab1, tab2 = st.tabs(["💨 Детекция", "📊 Информация о модели"])

with tab1:
    st.title("💨 Детекция ветряных турбин")
    st.write("Загрузите фото, и наша модель автоматически найдёт ветряные турбины.")

    input_type = st.radio("Способ загрузки", ["Файл", "URL"], key="input_type")

    images_to_process = []

    if input_type == "Файл":
        uploaded_files = st.file_uploader(
            "Выберите изображения (можно несколько)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="file_uploader"
        )
        if uploaded_files:
            for f in uploaded_files:
                try:
                    pil_img = Image.open(f).convert("RGB")
                    images_to_process.append((f.name, pil_img))
                except Exception as e:
                    st.error(f"Не удалось открыть {f.name}: {e}")

    elif input_type == "URL":
        urls_text = st.text_area(
            "Введите URL изображений (по одному на строку)", 
            height=100,
            key="url_input"
        )
        if urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    images_to_process.append((f"url_{i+1}.jpg", image))
                except Exception as e:
                    st.error(f"Ошибка загрузки {url}: {e}")

    # ----------------------------
    # ОБРАБОТКА ИЗОБРАЖЕНИЙ
    # ----------------------------
    if images_to_process and model is not None:
        st.subheader(f"Результаты ({len(images_to_process)} изображений)")
        
        for idx, (name, pil_img) in enumerate(images_to_process):
            st.image(pil_img, caption=f"Оригинал: {name}", width=700)
            
            with st.spinner("Детекция турбин..."):
                # Конвертируем PIL → OpenCV (BGR)
                img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Запускаем предсказание YOLO
                start_time = time.time()
                results = model(img_cv)
                elapsed = time.time() - start_time
                
                # Рисуем результат на изображении
                annotated_img = results[0].plot()  # возвращает BGR numpy array
                
                # Конвертируем обратно в RGB для PIL
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            st.image(annotated_pil, caption=f"Результат: {name}", width=700)
            st.caption(f"⏱️ Время предсказания: {elapsed:.2f} сек")

    elif images_to_process:
        st.error("Модель не загружена — обработка невозможна")

# ----------------------------
# ИНФОРМАЦИЯ О МОДЕЛИ
# ----------------------------
with tab2:
    st.header("ℹ️ Информация о модели")
    model_dir = 'metrics/YOLO_wind'
    
    if model:
        st.write("🔹 Модель: YOLO11x")
        st.write("🔹 Обучена на 3020 объектов")
        st.write("🔹 Число эпох обучения: 80")
        st.write("🔹 PR Curve: 0.80")

        # Графики
        graphs = [
            ("📈 Loss & Metrics", "results.png"),
            ("🎯 Precision-Recall", "BoxPR_curve.png"),
            ("🧩 Confusion Matrix", "confusion_matrix.png")
        ]

        for title, filename in graphs:
            st.subheader(title)
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                st.image(path, width=800)
            else:
                st.warning(f"Файл не найден: {filename}")
    else:
        st.error("Модель не загружена")

st.sidebar.title('Навигация 🧭')
st.sidebar.page_link('app.py', label='Forest Segmentation', icon='🌲')
st.sidebar.page_link('pages/face.py', label='Detector Face', icon='👁️')
st.sidebar.page_link('pages/sudno.py', label='Detector Ships', icon='⛴️')
st.sidebar.page_link('pages/wind.py', label='Detector Wind Turbines', icon='💨')