import streamlit as st

def show_login():
    st.header("Вход")
    username = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type="password")
    login_clicked = st.button("Войти")
    return username, password, login_clicked

def show_register():
    st.header("Регистрация")
    username = st.text_input("Имя пользователя", key="reg_username")
    password = st.text_input("Пароль", type="password", key="reg_password")
    confirm_password = st.text_input("Подтвердите пароль", type="password", key="reg_confirm")
    register_clicked = st.button("Зарегистрироваться")
    return username, password, confirm_password, register_clicked

def show_logout(username):
    st.write(f"Вы вошли как **{username}**")
    if st.button("Выйти"):
        return True
    return False

def show_error(message):
    st.error(message)

def show_success(message):
    st.success(message)

