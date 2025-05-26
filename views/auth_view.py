import streamlit as st

def show_login(disabled=False):
    st.header("Вход")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Имя пользователя", disabled=disabled)
        password = st.text_input("Пароль", type="password", disabled=disabled)
        submitted = st.form_submit_button("Войти", disabled=disabled)
    return username, password, submitted

def show_register(disabled=False):
    st.header("Регистрация")
    with st.form("register_form", clear_on_submit=False):
        username = st.text_input("Имя пользователя", disabled=disabled)
        password = st.text_input("Пароль", type="password", disabled=disabled)
        confirm_password = st.text_input("Подтвердите пароль", type="password", disabled=disabled)
        submitted = st.form_submit_button("Зарегистрироваться", disabled=disabled)
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


