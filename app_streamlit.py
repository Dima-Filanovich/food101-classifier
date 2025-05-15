import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import urllib.parse
from deep_translator import GoogleTranslator
from keras.layers import TFSMLayer  # ВАЖНО: новый способ загрузки SavedModel


# Загрузка модели
@st.cache_resource
def load_model():
    model = TFSMLayer("food101_modelon", call_endpoint="serving_default")
    return model

model = load_model()

# Классы
CLASS_NAMES = [  # (Оставил без изменений)
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

# Функция получения пищевой информации
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

# Интерфейс Streamlit
st.title("🍽️ Классификатор еды — Food101")
st.write("Загрузите изображение блюда, и модель определит его категорию.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    st.write("🔍 Распознавание...")
    img_batch = preprocess_image(image)
    img_tensor = tf.convert_to_tensor(img_batch)
    output_dict = model(img_tensor)                        # Возвращает dict
    predictions = list(output_dict.values())[0].numpy()[0] # Получаем тензор и превращаем в numpy


    # Топ-3 предсказания
    top_indices = predictions.argsort()[-3:][::-1]
    st.subheader("🔝 Топ-3 предсказания:")
    for i in top_indices:
        class_name = CLASS_NAMES[i].replace('_', ' ').title()
        confidence = predictions[i]
        st.write(f"{class_name}: {confidence:.2%}")


    # Основное предсказание
    predicted_class = CLASS_NAMES[top_indices[0]].replace('_', ' ').title()
    st.success(f"🍽️ Это скорее всего: **{predicted_class}** ({predictions[top_indices[0]]:.2%} уверенности)")

    # Получение информации о питательных веществах
    nutrition_info = get_nutrition_info(predicted_class)
    if nutrition_info:
        st.subheader("🧪 Пищевая ценность (на 100г):")
        st.write(f"**Калории:** {nutrition_info['energy_kcal']} ккал")
        st.write(f"**Белки:** {nutrition_info['proteins']} г")
        st.write(f"**Жиры:** {nutrition_info['fat']} г")
        st.write(f"**Углеводы:** {nutrition_info['carbohydrates']} г")

        # Название на русском
        product_name_ru = nutrition_info.get("product_name_ru")
        if not product_name_ru:
            try:
                product_name_ru = GoogleTranslator(source='en', target='ru').translate(predicted_class)
            except Exception:
                product_name_ru = "Перевод недоступен"
        st.write(f"**Название на русском:** {product_name_ru}")

        # Ссылка на источник
        product_url = nutrition_info.get("url")
        if product_url:
            st.markdown(f"[📎 Подробнее на Open Food Facts]({product_url})")
    else:
        st.warning("Информация о пищевой ценности не найдена.")
