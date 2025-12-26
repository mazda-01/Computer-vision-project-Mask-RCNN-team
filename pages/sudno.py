import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import pandas as pd


# ====== 1. КАСТОМНАЯ ТЕМА ЧЕРЕЗ HTML/CSS ======
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
        background-image: url("https://www.shutterstock.com/shutterstock/videos/746908/thumb/1.jpg?ip=x480");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no;
    }

    /* Тёмные карточки с белым текстом */
    .css-1v0mbdj, .css-12w0y3b, .stMarkdown, .stTabs, .stDataFrame, 
    .stPlotlyChart, .stImage, .stTable, div[data-testid="stHorizontalBlock"] {
        background-color: rgba(15, 23, 42, 0.88) !important;
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
        background-color: rgba(51, 65, 85, 0.6);
        border-radius: 10px 10px 0 0;
        color: #cbd5e1;
        font-weight: 600;
        padding: 0 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155;
        color: #e2e8f0;
    }

    /* Графики matplotlib — не выходят за границы */
    .stPlotlyChart, .stPyplot {
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title('Навигация 🧭')
st.sidebar.page_link('app.py', label='Forest Segmentation', icon='🌲')
st.sidebar.page_link('pages/face.py', label='Detector Face', icon='👁️')
st.sidebar.page_link('pages/sudno.py', label='Detector Ships', icon='⛴️')

# ====== 2. ЗАГОЛОВОК ======
st.title("🚢 Модель для детекции судов на изображениях аэросъёмки")

# ====== 3. ВКЛАДКИ ======
tabs = st.tabs([
    "📊  Датасет",
    "📈 Первое обучение",
    "🔄 Попытки улучшения",
    "🏆 Итоговая модель",
    "🔍 Детекция судов"
])

# ======================
# ВКЛАДКА 1: ДАТАСЕТ
# ======================
with tabs[0]:
    st.subheader("📦 Структура датасета")
    train_count, valid_count, test_count = 9697, 2165, 1573
    st.markdown(f"""
    Датасет состоит из изображений, разделённых на три части:
    - **Train**: {train_count} изображений
    - **Validation**: {valid_count} изображений
    - **Test**: {test_count} изображений
    """)

    # График
    labels = ['Train', 'Validation', 'Test']
    counts = [train_count, valid_count, test_count]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=['#3b82f6', '#10b981', '#ef4444'])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Количество изображений')
    ax.set_ylim(0, max(counts) * 1.1)
    st.pyplot(fig)

    # Примеры изображений
    st.subheader("🖼️ Примеры из train-выборки")
    sample_paths = [f"images/sudno/{i}.jpg" for i in range(1, 5)]
    cols = st.columns(4)
    for idx, col in enumerate(cols):
        img_path = Path(sample_paths[idx])
        if img_path.exists():
            img = Image.open(img_path)
            col.image(img, use_container_width=True, caption=f"Пример {idx+1}")
        else:
            col.warning("Файл не найден")

# ======================
# ВКЛАДКА 2: ПЕРВОЕ ОБУЧЕНИЕ
 #модель, начальные параметры, итог первого обучения: метрики, графики и примеры предсказаний
# ======================
with tabs[1]:
    st.subheader("📉 Модель YOLO11m")
    st.write('Результаты первых 10 эпох')
    
    try:
        df1 = pd.read_csv('metrics/YOLO_sudno/results_start_1.csv')
        df1 = df1.drop(['epoch', 'time'], axis=1, errors='ignore')
        df1.index = [f"{i} epoch" for i in range(1, 11)]
        st.dataframe(df1.style.format("{:.4f}"))
    except Exception as e:
        st.error("Не удалось загрузить метрики обучения.")
    
    # Кривые
    curve_files = {
        "Precision-Recall": "metrics/YOLO_sudno/pr_curve.png",
        "F1-Score": "metrics/YOLO_sudno/f1_curve.png",
        "Матрица ошибок": "metrics/YOLO_sudno/confusion_matrix.png",
        "Loss-кривая": "metrics/YOLO_sudno/loss_curve.png"
    }

    for title, path in curve_files.items():
        st.markdown(f"### {title}")
        p = Path(path)
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.info(f"График «{title}» недоступен.")

    # Примеры предсказаний
    st.subheader("🔍 Примеры предсказаний (1–10 эпох)")
    pred_files = [f"metrics/YOLO_sudno/pred_img{i}.jpg" for i in range(2, 5)]
    cols = st.columns(3)
    for i, col in enumerate(cols):
        p = Path(pred_files[i])
        if p.exists():
            col.image(str(p), use_container_width=True, caption=f"Предсказание {i+1}")
        else:
            col.warning("Изображение не найдено")

# ======================
# ВКЛАДКА 3: УЛУЧШЕНИЯ
# ======================
with tabs[2]:
    st.subheader("🔧 Стратегии улучшения модели")
    st.markdown("""
    - Увеличение числа эпох
    - Аугментация данных (Mosaic, mixup)
    - Настройка гиперпараметров (lr0, optimizer)
    """)
    
    st.subheader("📊 Сравнение метрик")   

    st.markdown("**Конец первого обучения**")
    try:
        row = df1.loc["10 epoch"]
        st.dataframe(row.to_frame().T)
    except:
        st.error("Данные не загружены")  

    st.markdown("**Второе обучение**")
    try:
        df2 = pd.read_csv('metrics/YOLO_sudno/results_start_2.csv')
        df2 = df2  .drop(['epoch', 'time'], axis=1)
        df2.index = [f"{i + 10} epoch" for i in range(1, 11)]
        st.dataframe(df2.style.format("{:.4f}"))
    except:
        st.error("Данные не загружены")


# ======================
# ВКЛАДКА 4: ИТОГОВАЯ МОДЕЛЬ
# ======================
# with tabs[3]:
    # st.subheader("🎖️ Характеристики финальной модели")
    # st.markdown("""
    # - **Архитектура**: YOLOv11m (кастомная модификация)
    # - **Класс**: `ship`
    # - **Итоговые метрики**:
    #   - **mAP@0.5**: 0.892
    #   - **mAP@0.5:0.95**: 0.674
    #   - **Precision**: 0.915
    #   - **Recall**: 0.871
    # - **Веса**: `models/best.pt`
    # """)
    # st.image("images/final_metrics.png", caption="Итоговые метрики", use_container_width=True)

# ======================
# ВКЛАДКА 5: ДЕТЕКЦИЯ
# ======================
with tabs[4]:
    st.subheader("🎯 Загрузите изображения для детекции судов")
    
    @st.cache_resource
    def load_model():
        return YOLO('models/sudno.pt')  # ← замените на best.pt

    model = load_model()

    uploaded_files = st.file_uploader(
        "Выберите изображения", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    url = st.text_input("Или вставьте URL изображения")

    images_to_process = []

    if uploaded_files:
        for f in uploaded_files:
            try:
                img = Image.open(f).convert("RGB")
                images_to_process.append((f"Файл: {f.name}", img))
            except Exception as e:
                st.error(f"Ошибка при открытии {f.name}: {e}")

    if url:
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            images_to_process.append(("Изображение по URL", img))
        except Exception as e:
            st.error(f"Ошибка загрузки по URL: {e}")

    if images_to_process:
        for label, image in images_to_process:
            st.markdown(f"### {label}")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            with st.spinner("Идёт детекция..."):
                results = model(image)
            
            plotted = results[0].plot()
            plotted_rgb = plotted[..., ::-1]
            st.image(plotted_rgb, caption="Результат детекции", use_container_width=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"✅ Обнаружено **{len(boxes)}** судно(а/ов):")
                for i, box in enumerate(boxes, 1):
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    name = model.names[cls_id] if hasattr(model, 'names') else 'ship'
                    st.markdown(f"**{i}.** `{name}` — уверенность: **{conf*100:.1f}%**")
            else:
                st.info("🧭 Судов не обнаружено.")
            st.divider()
   
