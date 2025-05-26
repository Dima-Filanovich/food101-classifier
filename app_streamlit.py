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

    if "user" not in st.session_state:
        st.session_state.user = None
    if "register_success" not in st.session_state:
        st.session_state.register_success = False

    if st.session_state.user is None:
        st.title("Добро пожаловать в Food101 Classifier")

        # Показываем сообщение о регистрации, если оно было успешно
        if st.session_state.register_success:
            st.success("✅ Регистрация прошла успешно! Пожалуйста, войдите.")
            st.session_state.register_success = False

        tab_login, tab_register = st.tabs(["Вход", "Регистрация"])

        with tab_login:
            username, password, login_clicked = show_login()
            if login_clicked:
                with st.spinner("⏳ Входим в систему..."):
                    try:
                        success, msg, user = auth_ctrl.login(username, password)
                        if success:
                            st.session_state.user = user
                            show_success("✅ Успешный вход!")
                            st.rerun()
                            st.stop()  # 🔒 Остановить выполнение
                        else:
                            show_error(msg)
                    except Exception as e:
                        show_error(f"❌ Ошибка при входе: {e}")

        with tab_register:
            username, password, confirm_password, register_clicked = show_register()
            if register_clicked:
                with st.spinner("⏳ Регистрируем пользователя..."):
                    try:
                        success, msg = auth_ctrl.register(username, password, confirm_password)
                        if success:
                            st.session_state.register_success = True
                            st.rerun()  # Перезапускаем, чтобы показать сообщение
                            st.stop()  # 🔒 Остановить выполнение
                        else:
                            show_error(msg)
                    except Exception as e:
                        show_error(f"❌ Ошибка при регистрации: {e}")
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
            # Показываем историю предсказаний
            history = predict_ctrl.get_history(user["id"])
            if history:
                with st.expander("🕘 История ваших предсказаний (последние 10):", expanded=False):
                    for item in history:
                        image_name = item['image_name']
                        if isinstance(image_name, bytes):
                            image_name = image_name.decode("utf-8")
                        
                        st.markdown(f"""
                        **📷 Изображение:** {item['image_name']}  
                        **🍽 Предсказание:** {item['predicted_class']}  
                        **✅ Уверенность:** {item['confidence']:.2%}  
                        **🕒 Дата:** {item['timestamp']}  
                        ---
                        """)
            else:
                st.info("История пуста.")



if __name__ == "__main__":
    init_db()
    main()
