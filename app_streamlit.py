import streamlit as st
from models.database import init_db

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

    # Состояния сессии
    for key in [
        "user", "register_success", "login_clicked", "register_clicked", 
        "is_loading", "login_username", "login_password", "register_data", 
        "auth_mode"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None if key == "user" else False

    st.session_state.auth_mode = st.session_state.auth_mode or "Вход"

    if st.session_state.user is None:
        st.title("Добро пожаловать в Food101 Classifier")

        if st.session_state.register_success:
            show_success("✅ Регистрация прошла успешно! Пожалуйста, войдите.")
            st.session_state.register_success = False

        # Радио-переключатель с блокировкой при загрузке
        st.session_state.auth_mode = st.radio(
            "Выберите действие", ["Вход", "Регистрация"],
            index=0 if st.session_state.auth_mode == "Вход" else 1,
            horizontal=True,
            disabled=st.session_state.is_loading
        )

        if st.session_state.is_loading:
            st.info("⏳ Пожалуйста, подождите...")

        if st.session_state.auth_mode == "Вход":
            username, password, login_clicked = show_login()
            if login_clicked and not st.session_state.is_loading:
                st.session_state.login_clicked = True
                st.session_state.login_username = username
                st.session_state.login_password = password
                st.session_state.is_loading = True
                st.rerun()

        elif st.session_state.auth_mode == "Регистрация":
            username, password, confirm_password, register_clicked = show_register()
            if register_clicked and not st.session_state.is_loading:
                st.session_state.register_clicked = True
                st.session_state.register_data = (username, password, confirm_password)
                st.session_state.is_loading = True
                st.rerun()

        if st.session_state.login_clicked:
            with st.spinner("⏳ Входим в систему..."):
                try:
                    success, msg, user = auth_ctrl.login(
                        st.session_state.login_username,
                        st.session_state.login_password
                    )
                    if success:
                        st.session_state.user = user
                        show_success("✅ Успешный вход!")
                        st.session_state.login_clicked = False
                        st.rerun()
                    else:
                        show_error(msg)
                except Exception as e:
                    show_error(f"❌ Ошибка при входе: {e}")
                finally:
                    st.session_state.is_loading = False

        if st.session_state.register_clicked:
            with st.spinner("⏳ Регистрируем пользователя..."):
                try:
                    username, password, confirm_password = st.session_state.register_data
                    success, msg = auth_ctrl.register(username, password, confirm_password)
                    if success:
                        st.session_state.register_success = True
                        st.session_state.register_clicked = False
                        st.rerun()
                    else:
                        show_error(msg)
                except Exception as e:
                    show_error(f"❌ Ошибка при регистрации: {e}")
                finally:
                    st.session_state.is_loading = False

    else:
        user = st.session_state.user
        if show_logout(user["username"]):
            st.session_state.user = None
            st.rerun()

        uploaded_file = show_upload_section()

        if uploaded_file:
            image = predict_ctrl.load_image(uploaded_file)
            show_image(image)

            top_classes, confidences = predict_ctrl.predict(image)
            show_predictions(top_classes, confidences)
            show_prediction_result(top_classes[0], confidences[0])

            with st.spinner("⏳ Получение информации о пищевой ценности..."):
                nutrition_info = nutrition_ctrl.get_nutrition_info(top_classes[0])
                product_name_ru = nutrition_ctrl.translate_if_needed(nutrition_info, top_classes[0])

            if nutrition_info:
                show_nutrition_info(nutrition_info, top_classes[0], product_name_ru)
                report = predict_ctrl.make_report(
                    predicted_class=top_classes[0],
                    confidence=confidences[0],
                    nutrition_info=nutrition_info,
                )
                show_download_report(report)
            else:
                show_no_nutrition_warning()

            predict_ctrl.save_history(user["id"], top_classes[0], confidences[0], uploaded_file.name)

            history = predict_ctrl.get_history(user["id"])
            for item in history:
                image_name = item['image_name']
                predicted_class = item['predicted_class']
                confidence = item['confidence']
                timestamp = item['timestamp']

                if isinstance(image_name, bytes):
                    image_name = image_name.decode("utf-8")
                if isinstance(predicted_class, bytes):
                    predicted_class = predicted_class.decode("utf-8")
                if isinstance(confidence, bytes):
                    try:
                        confidence = float(confidence.decode("utf-8"))
                    except Exception:
                        confidence = 0.0
                elif not isinstance(confidence, float):
                    try:
                        confidence = float(confidence)
                    except Exception:
                        confidence = 0.0
                if isinstance(timestamp, bytes):
                    timestamp = timestamp.decode("utf-8")

                st.markdown(f"""
                **📷 Изображение:** {image_name}  
                **🍽 Предсказание:** {predicted_class}  
                **✅ Уверенность:** {confidence:.2%}  
                **🕒 Дата:** {timestamp}  
                ---""")

if __name__ == "__main__":
    init_db()
    main()



