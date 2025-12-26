import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import pandas as pd

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
        background-image: url("https://balthazar.club/o/uploads/posts/2024-01/1705040959_balthazar-club-p-krasivii-fon-lesa-oboi-46.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no;
    }

    /* Тёмные карточки с белым текстом */
    .css-1v0mbdj, .css-12w0y3b, .stMarkdown, .stTabs, .stDataFrame, 
    .stPlotlyChart, .stImage, .stTable, div[data-testid="stHorizontalBlock"] {
        background-color: rgba(35, 54, 35, 0.88) !important;
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
        background-color: rgba(35, 54, 35, 0.6);
        border-radius: 10px 10px 0 0;
        color: #cbd5e1;
        font-weight: 600;
        padding: 0 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #345533;
        color: #e2e8f0;
    }

    /* Графики matplotlib — не выходят за границы */
    .stPlotlyChart, .stPyplot {
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Настройка страницы (тёмная тема по умолчанию — красивее для масок)
st.set_page_config(page_title="Сегментация леса", layout="centered")

# Вкладки
tab1, tab2 = st.tabs(["🌲 Сегментация", "📊 Информация о модели"])

with tab1:
    st.title("🌲 Детектор леса на аэрофотоснимках")
    st.markdown("Загружайте фото или вставьте ссылку — модель покажет, где лес.")

    MODEL_PATH = "models/best_unet.pth"

    @st.cache_resource
    def load_model():
        if not os.path.exists(MODEL_PATH):
            st.error(f"Модель не найдена! Положите '{MODEL_PATH}' рядом с app.py")
            return None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device

    model, device = load_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def segment_image(image_pil):
        if model is None:
            return None, 0.0, 0.0
    
        img_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output)[0, 0].cpu().numpy()


        orig_w, orig_h = image_pil.size

        # Выбираем порог в зависимости от размера
        if orig_w > 800 and orig_h > 800:
            threshold = 0.6
        else:
            threshold = 0.3
        
        mask = (probs > threshold).astype(np.uint8) * 255
        forest_percent = (probs > threshold).mean() * 100
        confidence = probs.mean() * 100
        return mask, forest_percent, confidence

    # === Загрузка файлов ===
    st.header("Загрузите изображения")
    uploaded_files = st.file_uploader(
        "Выберите фото (JPG/PNG)", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")

            # Оригинал
            st.image(image, caption=f"Оригинал: {uploaded_file.name}", width=700)

            # Сегментация ниже
            mask, forest_percent, confidence = segment_image(image)

            if mask is not None:
                # Настройка matplotlib — убираем белый фон и оси
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.imshow(mask, cmap="Greens", alpha=0.6)
                plt.axis('off')
                plt.margins(0, 0)
                plt.tight_layout(pad=0)

                st.pyplot(plt, use_container_width=True)  # без белого окна

                st.success("**Результат:**")
                st.write(f"🌲 Лес занимает **{forest_percent:.1f}%** площади")
                st.write(f"📊 Уверенность модели: **{confidence:.1f}%**")
                st.markdown("---")  # разделитель между изображениями

    # === По URL ===
    st.header("Вставьте ссылку на фото")
    url = st.text_input("Прямая ссылка:")

    if url:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

            st.image(image, caption="Оригинал по ссылке", width=700)

            mask, forest_percent, confidence = segment_image(image)

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(mask, cmap="Greens", alpha=0.6)
            plt.axis('off')
            plt.margins(0, 0)
            plt.tight_layout(pad=0)

            st.pyplot(plt, use_container_width=True)

            st.success("**Результат:**")
            st.write(f"🌲 Лес занимает **{forest_percent:.1f}%** площади")
            st.write(f"📊 Уверенность модели: **{confidence:.1f}%**")

        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

with tab2:
    st.header("📊 Информация о модели")

    st.write("**Параметры:**")
    st.write("🔹 Архитектура: SMP UNet (EfficientNet-B4 backbone)")
    st.write("🔹 Эпох обучения: 20")
    st.write("🔹 Обучена на 5100 объектов")
    st.write("🔹 Loss: BCEWithLogitsLoss")
    st.write("🔹 PR AUC: 0.94")

    metrics_csv = "metrics/training_metrics.csv"
    metrics_png = "metrics/training_plots.png"

    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        st.subheader("Метрики по эпохам")
        st.dataframe(df.style.format("{:.4f}"))

    if os.path.exists(metrics_png):
        st.subheader("Графики обучения")
        st.image(Image.open(metrics_png), width=1000)

st.caption("Демонстрация модели сегментации леса на аэрофотоснимках.")

st.sidebar.title('Навигация 🧭')
st.sidebar.page_link('app.py', label='Forest Segmentation', icon='🌲')
st.sidebar.page_link('pages/face.py', label='Detector Face', icon='👁️')
st.sidebar.page_link('pages/sudno.py', label='Detector Ships', icon='⛴️')
# st.sidebar.page_link('pages/analysis.py', label='Анализ', icon='📊')