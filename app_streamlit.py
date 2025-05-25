import streamlit as st
from models.database import init_db  # <-- импорт

from controllers.auth_controller import AuthController
from controllers.predict_controller import PredictController
from controllers.nutrition_controller import NutritionController

from views.auth_view import show_login, show_register, show_logout, show_error, show_success
from views.prediction_view import (
    show_upload_section, show_image, show_predictions, show_prediction_result,
    show_nutrition_info, show_download_report, show_no_nutrition_warning
)

def main():
    init_db()
    st.set_page_config(page_title="Food101 Classifier", page_icon="🍽️")

    auth_ctrl = AuthController()
    predict_ctrl = PredictController()
    nutrition_ctrl = NutritionController()

    # Сессия для хранения состояния
    if "user" not in st.session_state:
        st.session_state.user = None

    # Если пользователь не залогинен — показать форму входа/регистрации
    if st.session_state.user is None:
        st.title("Добро пожаловать в Food101 Classifier")
        tab = st.tabs(["Вход", "Регистрация"])
        
        with tab[0]:
            username, password, login_clicked = show_login()
            if login_clicked:
                success, msg, user = auth_ctrl.login(username, password)
                if success:
                    st.session_state.user = user
                    show_success("Успешный вход!")
                    st.experimental_rerun()
                else:
                    show_error(msg)
        
        with tab[1]:
            username, password, confirm_password, register_clicked = show_register()
            if register_clicked:
                success, msg = auth_ctrl.register(username, password, confirm_password)
                if success:
                    show_success("Регистрация прошла успешно! Войдите в систему.")
                else:
                    show_error(msg)
    else:
        # Пользователь залогинен, показать интерфейс классификатора
        user = st.session_state.user
        if show_logout(user.username):
            st.session_state.user = None
            st.experimental_rerun()

        uploaded_file = show_upload_section()

        if uploaded_file:
            image = predict_ctrl.load_image(uploaded_file)
            show_image(image)

            top_classes, confidences = predict_ctrl.predict(image)
            show_predictions(top_classes, confidences)

            show_prediction_result(top_classes[0], confidences[0])

            # Получаем питание
            nutrition_info = nutrition_ctrl.get_nutrition(top_classes[0])

            # Перевод названия (если нужно)
            product_name_ru = nutrition_ctrl.translate_if_needed(nutrition_info, top_classes[0])

            if nutrition_info:
                show_nutrition_info(nutrition_info, top_classes[0], product_name_ru)

                report = predict_ctrl.make_report(
                    predicted_class=top_classes[0],
                    confidence=confidences[0],
                    nutrition_info=nutrition_info,
                    product_name_ru=product_name_ru
                )
                show_download_report(report)
            else:
                show_no_nutrition_warning()

            # Сохраняем историю в БД
            predict_ctrl.save_history(user.id, top_classes[0], confidences[0])

if __name__ == "__main__":
    main()

