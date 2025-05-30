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

# ВРЕМЕННОЕ ОТКЛЮЧЕНИЕ
MAINTENANCE_MODE = False
if MAINTENANCE_MODE:
    st.error("🚧 Приложение находится на обслуживании.")
    st.stop()

# Загрузка модели
@st.cache_resource
def load_model():
    return TFSMLayer("food101_modelon", call_endpoint="serving_default")

model = load_model()

@st.cache_data
def get_translated_classes():
    translations = {}
    for cls in CLASS_NAMES:
        readable_name = cls.replace("_", " ").title()
        try:
            ru_name = GoogleTranslator(source='en', target='ru').translate(readable_name)
        except Exception:
            ru_name = readable_name
        translations[cls] = ru_name
    return translations

CLASS_TRANSLATIONS = get_translated_classes()



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

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Улучшенный поиск информации о пище
def get_nutrition_info(food_name):
    def try_query(query):
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={urllib.parse.quote(query)}&search_simple=1&action=process&json=1&page_size=5"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for product in data.get("products", []):
                    nutriments = product.get("nutriments", {})
                    if nutriments.get("energy-kcal_100g"):
                        return {
                            "product_name": product.get("product_name", ""),
                            "product_name_ru": product.get("product_name_ru", ""),
                            "energy_kcal": nutriments.get("energy-kcal_100g"),
                            "proteins": nutriments.get("proteins_100g"),
                            "fat": nutriments.get("fat_100g"),
                            "carbohydrates": nutriments.get("carbohydrates_100g"),
                            "url": product.get("url", "")
                        }
        except Exception:
            return None
        return None

    base_query = food_name.replace("_", " ").lower()
    result = try_query(base_query)
    if not result and " " in base_query:
        result = try_query(base_query.split(" ")[0])
    return result

@st.cache_data(show_spinner=False)
def get_nutrition_info_cached(food_name):
    return get_nutrition_info(food_name)

# --- Интерфейс ---
st.title("🍽️ Классификатор еды — Food101")
st.write("Загрузите изображение блюда, и модель определит его категорию. Точность ~73%")
with st.expander("📖 Посмотреть все категории"):
    st.markdown("**Категории на английском и русском:**")
    cols = st.columns(2)
    half = len(CLASS_NAMES) // 2
    with cols[0]:
        for cls in CLASS_NAMES[:half]:
            st.markdown(f"`{cls.replace('_', ' ').title()}` → **{CLASS_TRANSLATIONS[cls]}**")
    with cols[1]:
        for cls in CLASS_NAMES[half:]:
            st.markdown(f"`{cls.replace('_', ' ').title()}` → **{CLASS_TRANSLATIONS[cls]}**")


uploaded_file = st.file_uploader("📤 Загрузите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Вы можете загрузить изображение. Вот пример:")
    example_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsfW388zWeoTBoYVtL5yJi85sJmFoVB3isLw&s"
    example_img = Image.open(requests.get(example_url, stream=True).raw).convert("RGB")
    st.image(example_img, caption="Пример: Хот-дог", use_container_width=True)

    img_byte_arr = io.BytesIO()
    example_img.save(img_byte_arr, format='JPEG')
    st.download_button("📥 Скачать пример", data=img_byte_arr.getvalue(), file_name="example.jpg", mime="image/jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    st.write("🔍 Анализ изображения...")
    img_batch = preprocess_image(image)
    img_tensor = tf.convert_to_tensor(img_batch)
    with st.spinner("🔍 Распознавание..."):
        output_dict = model(img_tensor)
        predictions = list(output_dict.values())[0].numpy()[0]

    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = [CLASS_NAMES[i].replace('_', ' ').title() for i in top_indices]
    confidences = [predictions[i] for i in top_indices]

    st.subheader("🔝 Топ-3 предсказания:")
    for name, conf in zip(top_classes, confidences):
        st.write(f"{name}: {conf:.2%}")

    df = pd.DataFrame({"Блюдо": top_classes, "Уверенность": confidences})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Уверенность:Q", axis=alt.Axis(format=".0%")),
        y=alt.Y("Блюдо:N", sort='-x'),
        color=alt.Color("Блюдо:N", legend=None)
    ).properties(height=150)
    st.altair_chart(chart, use_container_width=True)

    predicted_class = top_classes[0]
    if confidences[0] < 0.5:
        st.warning(f"⚠️ Низкая уверенность ({confidences[0]:.2%}). Возможное блюдо: **{predicted_class}**")
    else:
        st.success(f"🍽️ Предсказание: **{predicted_class}** ({confidences[0]:.2%} уверенности)")

    # ИНИЦИАЛИЗАЦИЯ СЕССИИ
    if "retry_clicked" not in st.session_state:
        st.session_state.retry_clicked = False

    # Функция получения пищевой информации (улучшенная)
    def get_nutrition_info(food_name):
        query = urllib.parse.quote(food_name.lower())
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size=5"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("products"):
                for product in data["products"]:
                    nutriments = product.get("nutriments", {})
                    if "energy-kcal_100g" in nutriments:
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

    # КЭШ
    @st.cache_data(show_spinner=False)
    def get_nutrition_info_cached(food_name):
        return get_nutrition_info(food_name)

    # Получение инфо и вывод
    with st.spinner("⏳ Получение информации о пищевой ценности..."):
        nutrition_info = get_nutrition_info_cached(predicted_class)

    if not nutrition_info and " " in predicted_class:
        # Попытка по первой части (например, "Chicken" из "Chicken Wings")
        nutrition_info = get_nutrition_info_cached(predicted_class.split(" ")[0])

    if nutrition_info:
        st.subheader("🧪 Пищевая ценность (на 100г):")
        st.write(f"**Калории:** {nutrition_info['energy_kcal']} ккал")
        st.write(f"**Белки:** {nutrition_info['proteins']} г")
        st.write(f"**Жиры:** {nutrition_info['fat']} г")
        st.write(f"**Углеводы:** {nutrition_info['carbohydrates']} г")

                # Визуализация БЖУ
        try:
            st.subheader("📊 Состав БЖУ (на 100г):")
            bju_data = pd.DataFrame({
                "Компонент": ["Белки", "Жиры", "Углеводы"],
                "Количество": [
                    float(nutrition_info['proteins'] or 0),
                    float(nutrition_info['fat'] or 0),
                    float(nutrition_info['carbohydrates'] or 0)
                ]
            })
            pie_chart = alt.Chart(bju_data).mark_arc(innerRadius=50).encode(
                theta="Количество:Q",
                color="Компонент:N",
                tooltip=["Компонент", "Количество"]
            ).properties(width=300, height=300)
            st.altair_chart(pie_chart, use_container_width=False)
        except Exception as e:
            st.info("⚠️ Не удалось построить график БЖУ.")


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
   







