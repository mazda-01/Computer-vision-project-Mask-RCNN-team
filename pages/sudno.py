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
st.sidebar.page_link('pages/wind.py', label='Detector Wind Turbines', icon='💨')

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
    pred_files = [f"metrics/YOLO_sudno/first_predictions/pred_img{i}.jpg" for i in range(1, 5)]
    cols = st.columns(4)
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

    st.subheader("🔧 Стратегии улучшения модели")
    st.markdown("""
    - Увеличение числа эпох
    - Аугментация данных (Mosaic, mixup)
    - Настройка гиперпараметров (lr0, optimizer)
    - Обучение с заморозкой слоев
    """)

    # Стилизация
    st.markdown("""
    <style>
        .metric-row {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #28a745;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1rem;
            width: 120px;
        }
        .metric-values {
            flex-grow: 1;
            text-align: center;
            font-size: 1.1rem;
        }
        .old-value {
            color: #6c757d;
        }
        .new-value {
            color: #28a745;
            font-weight: 700;
        }
        .arrow {
            margin: 0 10px;
            color: #495057;
        }
        .difference {
            color: #28a745;
            font-weight: 600;
            margin-left: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Данные
    metrics = [
        {"name": "Precision", "old": 0.508, "new": 0.564},
        {"name": "Recall", "old": 0.385, "new": 0.473},
        {"name": "mAP50", "old": 0.4, "new": 0.469},
        {"name": "mAP50-95", "old": 0.226, "new": 0.282}
    ]

    # Заголовок
    st.title("📊 Изменение метрик модели")
    st.markdown("---")

    # Отображение метрик
    for metric in metrics:
        difference = metric["new"] - metric["old"]
        percent_diff = (difference / metric["old"]) * 100
        
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-name">{metric['name']}</div>
            <div class="metric-values">
                <span class="old-value">{metric['old']:.3f}</span>
                <span class="arrow">→</span>
                <span class="new-value">{metric['new']:.3f}</span>
                <span class="difference">(+{difference:.3f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Итог
    st.markdown("---")
    st.success("✅ Все метрики показывают положительную динамику")

    


# ======================
# ВКЛАДКА 4: ИТОГОВАЯ МОДЕЛЬ
# ======================
with tabs[3]:
    st.subheader("Итоговая модель - базовая YOLOv8n на 30 эпохах 🤯")

    # Стилизация
    st.markdown("""
    <style>
        .metric-row {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1rem;
            width: 120px;
        }
        .metric-values {
            flex-grow: 1;
            text-align: center;
            font-size: 1.1rem;
        }
        .old-value {
            color: #6c757d;
        }
        .new-value {
            font-weight: 700;
        }
        .arrow {
            margin: 0 10px;
            color: #495057;
        }
        .difference {
            font-weight: 600;
            margin-left: 10px;
        }
        .positive {
            color: #28a745;
            border-left: 4px solid #28a745;
        }
        .negative {
            color: #dc3545;
            border-left: 4px solid #dc3545;
        }
    </style>
    """, unsafe_allow_html=True)

    # Данные
    metrics = [
        {"name": "Precision", "old": 0.566, "new": 0.585},
        {"name": "Recall", "old": 0.476, "new": 0.453},
        {"name": "mAP50", "old": 0.471, "new": 0.474},
        {"name": "mAP50-95", "old": 0.28, "new": 0.281}
    ]

    # Заголовок
    st.title("📊 Изменение метрик модели")
    st.markdown("---")
    # Информация о моделях
    st.markdown("""
    <div class="model-info">
        <span class="info-icon">ℹ️</span>
        <strong>Информация о моделях:</strong><br>
        • <strong>Серое значение</strong>: YOLOv11m с попытками улучшения каждые 10 эпох<br>
        • <strong>Цветное значение</strong>: Базовая YOLOv8n, обученная на 30 эпохах
    </div>
    """, unsafe_allow_html=True)

    # Отображение метрик
    for metric in metrics:
        difference = metric["new"] - metric["old"]
        percent_diff = (difference / metric["old"]) * 100
        
        # Определяем тип изменения (положительное/отрицательное)
        if difference >= 0:
            row_class = "positive"
            sign = "+"
            color_class = "positive"
        else:
            row_class = "negative"
            sign = ""
            color_class = "negative"
        
        st.markdown(f"""
        <div class="metric-row {row_class}">
            <div class="metric-name">{metric['name']}</div>
            <div class="metric-values">
                <span class="old-value">{metric['old']:.3f}</span>
                <span class="arrow">→</span>
                <span class="new-value {color_class}">{metric['new']:.3f}</span>
                <span class="difference {color_class}">({sign}{difference:.3f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Кривые
    curve_files = {
        "Матрица ошибок": "metrics/YOLO_sudno/last_confusion_matrix.png"}

    for title, path in curve_files.items():
        st.markdown(f"### {title}")
        p = Path(path)
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.info(f"График «{title}» недоступен.")

    st.title("🖼️ Сравнение предсказаний моделей")

    # Стили для красивого отображения
    st.markdown("""
    <style>
        .comparison-container {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            align-items: center;
        }
        .image-card {
            flex: 1;
            border-radius: 12px;
            padding: 15px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }
        .image-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: white;
        }
        .old-model {
            border-top: 4px solid #dc3545;
        }
        .new-model {
            border-top: 4px solid #28a745;
        }
    </style>
    """, unsafe_allow_html=True)

    # Пути к изображениям (замени на свои)
    old_images = [
        "metrics/YOLO_sudno/first_predictions/pred_img1.jpg",
        "metrics/YOLO_sudno/first_predictions/pred_img2.jpg",
        "metrics/YOLO_sudno/first_predictions/pred_img3.jpg",
        "metrics/YOLO_sudno/first_predictions/pred_img4.jpg"
    ]

    new_images = [
        "metrics/YOLO_sudno/last_predictions/pred_img1.jpg",
        "metrics/YOLO_sudno/last_predictions/pred_img2.jpg",
        "metrics/YOLO_sudno/last_predictions/pred_img3.jpg",
        "metrics/YOLO_sudno/last_predictions/pred_img4.jpg"
    ]

    # Сравнение изображений попарно
    for i, (old_img, new_img) in enumerate(zip(old_images, new_images), 1):
        st.markdown(f"### Пример #{i}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="image-card old-model">', unsafe_allow_html=True)
            st.markdown('<div class="image-title">YOLOv11m </div>', unsafe_allow_html=True)
            st.image(old_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="image-card new-model">', unsafe_allow_html=True)
            st.markdown('<div class="image-title">YOLOv8n </div>', unsafe_allow_html=True)
            st.image(new_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")  

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



   