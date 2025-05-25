import requests
import urllib.parse
from deep_translator import GoogleTranslator
import streamlit as st

class NutritionController:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_nutrition_info(self, food_name: str):
        query = urllib.parse.quote(food_name.lower())
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("products"):
                product = data["products"][0]
                nutriments = product.get("nutriments", {})
                return {
                    "product_name": product.get("product_name", ""),
                    "product_name_ru": product.get("product_name_ru", ""),
                    "energy_kcal": nutriments.get("energy-kcal_100g"),
                    "proteins": nutriments.get("proteins_100g"),
                    "fat": nutriments.get("fat_100g"),
                    "carbohydrates": nutriments.get("carbohydrates_100g"),
                    "url": product.get("url", "")
                }
        return None

    def translate_to_russian(self, text: str) -> str:
        try:
            return GoogleTranslator(source='en', target='ru').translate(text)
        except Exception:
            return "Перевод недоступен"
