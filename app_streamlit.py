import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import urllib.parse
from deep_translator import GoogleTranslator
from keras.layers import TFSMLayer
import pandas as pd
import altair as alt
import io

# ВРЕМЕННОЕ ОТКЛЮЧЕНИЕ ПРИЛОЖЕНИЯ
MAINTENANCE_MODE = False
if MAINTENANCE_MODE:
    st.error("🚧 Приложение находится на обслуживании. Возвращайтесь позже.")
    st.stop()

# Загрузка модели
@st.cache_resource
def load_model():
    return TFSMLayer("food101_modelon", call_endpoint="serving_default")

model = load_model()

# Список классов
CLASS_NAMES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]




# Предобработка изображения
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Получение информации о пище
def get_nutrition_info(food_name):
    query = urllib.parse.quote(food_name.lower())
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("products"):
            product = data["products"][0]
            nutriments = product.get("nutriments", {})
            return {
                "product_name": product.get("product_name", ""),
                "product_name_ru": product.get("product_name_ru", ""),
                "energy_kcal": nutriments.get("energy-kcal_100g"),
                "proteins": nutriments.get("proteins_100g"),
                "fat": nutriments.get("fat_100g"),
                "carbohydrates": nutriments.get("carbohydrates_100g"),
                "url": product.get("url", "")
            }
    return None

# Интерфейс

st.title("🍽️ Классификатор еды — Food101")
st.write("Загрузите изображение блюда, и модель определит его категорию.")
with st.expander("📖 Посмотреть все категории, которые распознаёт модель"):
    st.markdown(", ".join(f"`{c.replace('_', ' ').title()}`" for c in CLASS_NAMES))


uploaded_file = st.file_uploader("📤 Выберите изображение...", type=["jpg", "jpeg", "png"])

# Пример, если нет файла
if uploaded_file is None:
    st.info("Вы можете загрузить изображение. Вот пример:")
    example_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsfW388zWeoTBoYVtL5yJi85sJmFoVB3isLw&s"
    example_img = Image.open(requests.get(example_url, stream=True).raw).convert("RGB")
    st.image(example_img, caption="Пример: Хот-дог", use_container_width=True)

    # Скачивание изображения
    img_byte_arr = io.BytesIO()
    example_img.save(img_byte_arr, format='JPEG')
    st.download_button(
        label="📥 Скачать пример изображения",
        data=img_byte_arr.getvalue(),
        file_name="example_hotdog.jpg",
        mime="image/jpeg"
    )

# Обработка изображения
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    st.write("🔍 Результат распознавания:")
    img_batch = preprocess_image(image)
    img_tensor = tf.convert_to_tensor(img_batch)
    with st.spinner("🔍 Анализ изображения..."):
    	output_dict = model(img_tensor)
    	predictions = list(output_dict.values())[0].numpy()[0]

    # Топ-3
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = [CLASS_NAMES[i].replace('_', ' ').title() for i in top_indices]
    confidences = [predictions[i] for i in top_indices]

    st.subheader("🔝 Топ-3 предсказания:")
    for name, conf in zip(top_classes, confidences):
        st.write(f"{name}: {conf:.2%}")

    # График
    df = pd.DataFrame({"Блюдо": top_classes, "Уверенность": confidences})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Уверенность:Q", axis=alt.Axis(format=".0%")),
        y=alt.Y("Блюдо:N", sort='-x'),
        color=alt.Color("Блюдо:N", legend=None)
    ).properties(height=150)
    st.altair_chart(chart, use_container_width=True)

    # Основной результат
    predicted_class = top_classes[0]
    st.success(f"🍽️ Это скорее всего: **{predicted_class}** ({confidences[0]:.2%} уверенности)")

    # КЭШИРОВАННАЯ загрузка пищевой ценности
    @st.cache_data(show_spinner=False)
    def get_nutrition_info_cached(food_name):
        return get_nutrition_info(food_name)

    # Пищевая информация с ожиданием
    with st.spinner("⏳ Получение информации о пищевой ценности..."):
        nutrition_info = get_nutrition_info_cached(predicted_class)

    if nutrition_info:
        st.subheader("🧪 Пищевая ценность (на 100г):")
        st.write(f"**Калории:** {nutrition_info['energy_kcal']} ккал")
        st.write(f"**Белки:** {nutrition_info['proteins']} г")
        st.write(f"**Жиры:** {nutrition_info['fat']} г")
        st.write(f"**Углеводы:** {nutrition_info['carbohydrates']} г")

        product_name_ru = nutrition_info.get("product_name_ru")
        if not product_name_ru:
            try:
                product_name_ru = GoogleTranslator(source='en', target='ru').translate(predicted_class)
            except Exception:
                product_name_ru = "Перевод недоступен"
        st.write(f"**Название на русском:** {product_name_ru}")

        if nutrition_info.get("url"):
            st.markdown(f"[📎 Подробнее на Open Food Facts]({nutrition_info['url']})")

        # Скачать отчёт
        report = f"""
Предсказанное блюдо: {predicted_class}
Уверенность: {confidences[0]:.2%}

Калории: {nutrition_info['energy_kcal']} ккал
Белки: {nutrition_info['proteins']} г
Жиры: {nutrition_info['fat']} г
Углеводы: {nutrition_info['carbohydrates']} г
Название на русском: {product_name_ru}
"""
        st.download_button(
            label="📥 Скачать отчёт",
            data=report,
            file_name="food_prediction_report.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Информация о пищевой ценности не найдена.")
