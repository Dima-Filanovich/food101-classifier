import streamlit as st

def show_login():
    st.header("Вход")
    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
    return username, password, submitted

def show_register():
    st.header("Регистрация")
    with st.form("register_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        confirm_password = st.text_input("Подтвердите пароль", type="password")
        submitted = st.form_submit_button("Зарегистрироваться")
    return username, password, confirm_password, submitted

def show_logout(username):
    st.write(f"Вы вошли как **{username}**")
    if st.button("Выйти"):
        return True
    return False

def show_error(message):
    st.error(message)

def show_success(message):
    st.success(message)

