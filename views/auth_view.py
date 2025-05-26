import streamlit as st

def show_login():
    st.header("Вход")
    is_loading = st.session_state.get("is_loading", False)

    with st.form("login_form"):
        username = st.text_input("Имя пользователя", disabled=is_loading)
        password = st.text_input("Пароль", type="password", disabled=is_loading)
        submitted = st.form_submit_button("Войти", disabled=is_loading)
    return username, password, submitted

def show_register():
    st.header("Регистрация")
    is_loading = st.session_state.get("is_loading", False)

    with st.form("register_form"):
        username = st.text_input("Имя пользователя", disabled=is_loading)
        password = st.text_input("Пароль", type="password", disabled=is_loading)
        confirm_password = st.text_input("Подтвердите пароль", type="password", disabled=is_loading)
        submitted = st.form_submit_button("Зарегистрироваться", disabled=is_loading)
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


