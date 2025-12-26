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
    
    /* Основной контейнер — НЕ на весь экран, а с ограничением */
    .block-container {
        max-width: 1300px !important;   /* ← ключевой параметр */
        padding: 2rem 2rem !important;
        margin: 0 auto;                 /* центрируем */
    }

    /* Фон страницы (например, море) */
    .stApp {
        background-image: url("https://allwebs.ru/images/2025/12/26/871e99bbff703305321f4398c2398332.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no;
    }

    /* Тёмные карточки с белым текстом */
    .css-1v0mbdj, .css-12w0y3b, .stMarkdown, .stTabs, .stDataFrame, 
    .stPlotlyChart, .stImage, .stTable, div[data-testid="stHorizontalBlock"] {
        background-color: rgba(20, 20, 20, 0.88) !important;
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
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px 10px 0 0;
        color: #cbd5e1;
        font-weight: 600;
        padding: 0 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(220, 220, 220, 0.85);  /* светло-серый с прозрачностью */
        color: #1f2937;                                /* тёмный текст для контраста */
    }

    /* Графики matplotlib — не выходят за границы */
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
    model_path = "models/face.pt"  
    try:
        model = YOLO(model_path)
        # st.success(f"✅ Модель загружена: {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Не удалось загрузить модель: {e}")
        return None

model = load_model()

# ----------------------------
# ФУНКЦИЯ ДЛЯ БЛЮРИНГА ЛИЦ
# ----------------------------
def blur_faces_in_image(image_np):
    """
    Принимает изображение в формате numpy (BGR), возвращает изображение с заблюренными лицами.
    """
    if model is None:
        return image_np

    results = model(image_np)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls.item())  
            conf = box.conf.item()  
            xyxy = box.xyxy.tolist()[0] 
            

            if cls == 0 and conf > 0.5: 
                x1, y1, x2, y2 = map(int, xyxy)
                
                face_region = image_np[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image_np[y1:y2, x1:x2] = blurred_face
    
    return image_np

# Настройка страницы (тёмная тема по умолчанию — красивее для масок)
st.set_page_config(page_title="Сегментация леса", layout="centered")

# Вкладки
tab1, tab2 = st.tabs(["👁️ Детекция", "📊 Информация о модели"])

with tab1:
    st.title("👁️ Блюр лиц")
    st.write(
        "**Заблюрьте лица на фотографиях — защитите приватность!** "
        "Загрузите фото, и наша модель автоматически найдёт и размоет все лица."
    )

    # ----------------------------
    # БЛОК ЗАГРУЗКИ ИЗОБРАЖЕНИЙ
    # ----------------------------
    st.header("🖼️ Загрузите изображения")

    input_type = st.radio("Способ загрузки", ["Файл", "URL", "Веб-камера"], key="blur_input")

    images_to_process = []

    if input_type == "Файл":
        uploaded_files = st.file_uploader(
            "Выберите изображения (можно несколько)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="blur_file"
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
            key="blur_url"
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

    else:  # Веб-камера
        st.info("👉 Нажмите кнопку ниже, чтобы сделать фото с веб-камеры")
        camera_image = st.camera_input("Сделайте фото", key="camera_input")
        if camera_image is not None:
            try:
                pil_img = Image.open(camera_image).convert("RGB")
                images_to_process.append(("webcam.jpg", pil_img))
            except Exception as e:
                st.error(f"Не удалось обработать фото с камеры: {e}")

    # ----------------------------
    # ОБРАБОТКА ИЗОБРАЖЕНИЙ
    # ----------------------------
    if images_to_process and model is not None:
        st.subheader(f"Результаты ({len(images_to_process)} изображений)")
        
        n_cols = min(3, len(images_to_process))
        cols = st.columns(n_cols)
        
        for idx, (name, pil_img) in enumerate(images_to_process):
            with cols[idx % n_cols]:
                st.image(pil_img, caption=name, width=700)
                with st.spinner("Обработка..."):
                    start_time = time.time()
                    
                    # Конвертируем PIL → OpenCV (BGR)
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
                    # Блюрим лица
                    blurred_img_cv = blur_faces_in_image(img_cv)
                    
                    # Конвертируем обратно в PIL (RGB)
                    blurred_img_pil = Image.fromarray(cv2.cvtColor(blurred_img_cv, cv2.COLOR_BGR2RGB))
                    
                    elapsed = time.time() - start_time
                
                st.image(blurred_img_pil, caption=f"{name} (заблюрено)", width=700)
                st.caption(f"Время: {elapsed:.2f} сек")

    elif images_to_process:
        st.error("Модель не загружена — обработка невозможна")

# ----------------------------
# ИНФОРМАЦИЯ О МОДЕЛИ
# ----------------------------
with tab2:
    st.header("ℹ️ Информация о модели")
    model_dir = 'metrics/YOLO_face'
    if model:
        st.write("🔹 Модель: YOLOv8n")
        st.write("🔹 Обучена на 16 800 объектах")
        st.write("🔹 Число эпох обучения: 20")
        st.write("🔹 mean Average Precision: 0.88")
        st.write("🔹 Precision: 0.90")
        st.write("🔹 Recall: 0.80")
        st.write("🔹 F1-Score: 0.85")



        # Загружаем и показываем графики
        # Все графики в одну колонку (без накладывания)
        col1, = st.columns(1)  # ← ЗАПЯТАЯ ВАЖНА!

        with col1:
            st.subheader("📈 Loss & Metrics")
            if os.path.exists(os.path.join(model_dir, "results.png")):
                st.image(os.path.join(model_dir, "results.png"), caption="Общие метрики", width=800)
            else:
                st.warning("График results.png не найден")

        with col1:  # ← можно использовать ту же колонку, но лучше — каждый блок отдельно
            st.subheader("🎯 Precision-Recall")
            if os.path.exists(os.path.join(model_dir, "BoxPR_curve.png")):
                st.image(os.path.join(model_dir, "BoxPR_curve.png"), caption="Precision-Recall", width=800)
            else:
                st.warning("График BoxPR_curve.png не найден")
        with col1:  # ← можно использовать ту же колонку, но лучше — каждый блок отдельно
            st.subheader("🎯 F1-Confidence Curve")
            if os.path.exists(os.path.join(model_dir, "BoxF1_curve.png")):
                st.image(os.path.join(model_dir, "BoxF1_curve.png"), caption="F1-Confidence Curve", width=800)
            else:
                st.warning("График BoxF1_curve.png не найден")
        with col1:
            st.subheader("🧩 Confusion Matrix")
            if os.path.exists(os.path.join(model_dir, "confusion_matrix.png")):
                st.image(os.path.join(model_dir, "confusion_matrix.png"), caption="Матрица ошибок", width=800)
            else:
                st.warning("Матрица ошибок не найдена")
        with col1:
            st.subheader("Пример из валидационной выборки, True")
            if os.path.exists(os.path.join('images/face', "val_batch2_labels.jpg")):
                st.image(os.path.join('images/face', "val_batch2_labels.jpg"), caption="Пример из валидационной выборки, True", width=800)
            else:
                st.warning("Пример из валидационной выборки, True не найдена")      
        with col1:
            st.subheader("Пример из валидационной выборки, Predict")
            if os.path.exists(os.path.join('images/face', "val_batch2_pred.jpg")):
                st.image(os.path.join('images/face', "val_batch2_pred.jpg"), caption="Пример из валидационной выборки, Predict", width=800)
            else:
                st.warning("Пример из валидационной выборки, Predict не найдена")        
  

    else:
        st.error(f"❌ Папка модели не найдена: {model_dir}")
        st.write("Проверьте путь к модели или запустите обучение снова.")

st.sidebar.title('Навигация 🧭')
st.sidebar.page_link('app.py', label='Forest Segmentation', icon='🌲')
st.sidebar.page_link('pages/face.py', label='Detector Face', icon='👁️')
st.sidebar.page_link('pages/sudno.py', label='Detector Ships', icon='⛴️')
st.sidebar.page_link('pages/wind.py', label='Detector Wind Turbines', icon='💨')

