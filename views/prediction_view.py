import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

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


def show_upload_section():
    st.title("🍽️ Классификатор еды — Food101")
    st.write("Загрузите изображение блюда, и модель определит его категорию. Точность модели ~73%")
    with st.expander("📖 Посмотреть все категории, которые распознаёт модель"):
        st.markdown(", ".join(f"`{c.replace('_', ' ').title()}`" for c in CLASS_NAMES))

    uploaded_file = st.file_uploader("📤 Выберите изображение...", type=["jpg", "jpeg", "png"])
    return uploaded_file

def show_image(image: Image.Image):
    st.image(image, caption="Загруженное изображение", use_container_width=True)

def show_predictions(top_classes, confidences):
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

def show_prediction_result(predicted_class, confidence):
    if confidence < 0.5:
        st.warning(f"⚠️ Модель не уверена в распознавании (уверенность: {confidence:.2%}). Возможно, изображение не соответствует ни одной из категорий точно. Предположение: **{predicted_class}**")
    else:
        st.success(f"🍽️ Это скорее всего: **{predicted_class}** ({confidence:.2%} уверенности)")

def show_nutrition_info(nutrition_info, predicted_class, product_name_ru):
    st.subheader("🧪 Пищевая ценность (на 100г):")
    st.write(f"**Калории:** {nutrition_info['energy_kcal']} ккал")
    st.write(f"**Белки:** {nutrition_info['proteins']} г")
    st.write(f"**Жиры:** {nutrition_info['fat']} г")
    st.write(f"**Углеводы:** {nutrition_info['carbohydrates']} г")

    st.write(f"**Название на русском:** {product_name_ru}")

    if nutrition_info.get("url"):
        st.markdown(f"[📎 Подробнее на Open Food Facts]({nutrition_info['url']})")

def show_download_report(report):
    st.download_button(
        label="📥 Скачать отчёт",
        data=report,
        file_name="food_prediction_report.pdf",
        mime="application/pdf"
    )

def show_no_nutrition_warning():
    st.warning("⚠️ Информация о пищевой ценности не найдена.")

