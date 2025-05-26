import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import TFSMLayer
from models.history_model import add_history
import functools
from fpdf import FPDF
import tempfile
import os
import io

class PredictController:
    def __init__(self):
        self.model = self._load_model()
        self.CLASS_NAMES = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
            'waffles'
        ]

    @functools.lru_cache(maxsize=1)
    def _load_model(self):
        return TFSMLayer("food101_modelon", call_endpoint="serving_default")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def load_image(self, uploaded_file):
        """Загружает изображение из файла"""
        return Image.open(uploaded_file).convert("RGB")

    def predict(self, image: Image.Image, user_id: int = None, image_name: str = ""):
        img_tensor = tf.convert_to_tensor(self.preprocess_image(image))
        output_dict = self.model(img_tensor)
        predictions = list(output_dict.values())[0].numpy()[0]

        top_indices = predictions.argsort()[-3:][::-1]
        top_classes = [self.CLASS_NAMES[i].replace("_", " ").title() for i in top_indices]
        confidences = [predictions[i] for i in top_indices]

        if user_id:
            add_history(user_id, image_name, top_classes[0], confidences[0])

        return top_classes, confidences

    def predict_verbose(self, image: Image.Image, user_id: int = None, image_name: str = ""):
        """Упрощённый метод для UI: возвращает результат + описание + флаг уверенности"""
        top_classes, confidences = self.predict(image, user_id, image_name)
        predicted_class = top_classes[0]
        confidence = confidences[0]

        if confidence < 0.5:
            summary = f"⚠️ Модель не уверена (уверенность: {confidence:.2%}). Возможное блюдо: **{predicted_class}**"
            is_confident = False
        else:
            summary = f"🍽️ Это скорее всего: **{predicted_class}** ({confidence:.2%} уверенности)"
            is_confident = True

        return {
            "top_classes": top_classes,
            "confidences": confidences,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "summary": summary,
            "is_confident": is_confident
        }

    def make_report(self, predicted_class, confidence, nutrition_info):
        pdf = FPDF()
        pdf.add_page()

        # Подключение TTF-шрифта с поддержкой Unicode
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        pdf.cell(200, 10, txt=f"Класс: {predicted_class}", ln=True)
        pdf.cell(200, 10, txt=f"Уверенность: {confidence:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Название продукта: {nutrition_info.get('product_name')}", ln=True)
        
        pdf.cell(200, 10, txt=f"Калории: {nutrition_info.get('energy_kcal')} ккал", ln=True)
        pdf.cell(200, 10, txt=f"Белки: {nutrition_info.get('proteins')} г", ln=True)
        pdf.cell(200, 10, txt=f"Жиры: {nutrition_info.get('fat')} г", ln=True)
        pdf.cell(200, 10, txt=f"Углеводы: {nutrition_info.get('carbohydrates')} г", ln=True)
        pdf.cell(200, 10, txt=f"Источник: {nutrition_info.get('url')}", ln=True)

        # Сохраняем PDF в байтовый поток
        pdf_data = pdf.output(dest="S").encode("latin-1")  # FPDF возвращает строку, надо закодировать
        return io.BytesIO(pdf_data)

        
    def save_history(self, user_id: int, prediction: str, confidence: float, image_name: str = "uploaded_image.jpg"):
        """
        Сохраняет историю запроса пользователя в базу данных.

        :param user_id: ID пользователя
        :param prediction: Название предсказанного класса
        :param confidence: Уверенность модели
        :param image_name: Имя файла изображения (по умолчанию: uploaded_image.jpg)
        """
        try:
            add_history(user_id, image_name, prediction, confidence)
        except Exception as e:
            print(f"Ошибка при сохранении истории: {e}")

