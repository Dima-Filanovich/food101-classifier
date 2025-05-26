import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

def show_upload_section():
    st.title("🍽️ Классификатор еды — Food101")
    st.write("Загрузите изображение блюда, и модель определит его категорию. Точность модели ~73%")
    with st.expander("📖 Посмотреть все категории, которые распознаёт модель"):
        # Вызовите класс list из контроллера, здесь нужно будет передать список из контроллера
        pass

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

