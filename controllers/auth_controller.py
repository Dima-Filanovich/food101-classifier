import hashlib
from models.user_model import create_user, get_user

class AuthController:
    def __init__(self):
        pass

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str, confirm_password: str):
        if not username or not password or not confirm_password:
            return False, "Пожалуйста, заполните все поля"

        if password != confirm_password:
            return False, "Пароли не совпадают"

        try:
            if get_user(username):
                return False, "Пользователь с таким именем уже существует"
        except Exception as e:
            return False, f"Ошибка при проверке пользователя: {e}"

    password_hash = self.hash_password(password)
    try:
        success = create_user(username, password_hash)
        if success:
            return True, "Пользователь успешно зарегистрирован"
        else:
            return False, "Ошибка при регистрации пользователя"
    except Exception as e:
        return False, f"Ошибка базы данных при регистрации: {e}"

    def login(self, username: str, password: str):
        if not username or not password:
            return False, "Пожалуйста, заполните все поля", None

        user = get_user(username)
        if user and user[2] == self.hash_password(password):
            user_data = {"id": user[0], "username": user[1]}
            return True, "", user_data
        else:
            return False, "Неверное имя пользователя или пароль", None
