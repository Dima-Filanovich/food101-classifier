import hashlib
from models.user_model import create_user, get_user

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    password_hash = hash_password(password)
    return create_user(username, password_hash)

def login_user(username, password):
    user = get_user(username)
    if user and user[2] == hash_password(password):
        return {"id": user[0], "username": user[1]}
    return None
