import pandas as pd
import re

# Параметры обработки
# excluded_theme = ['tsitati']

# Чтение CSV-файла
df = pd.read_csv('jokes.csv')

# Фильтрация тем
# filterd_df = df[~df['theme'].isin(excluded_theme)]

df = df[df['rating'] >= 0]  # Оставляем шутки с рейтингом 3 и выше

# Функция для очистки текста
def clean_text(text):
    # Удаляем переносы строк и лишние пробелы
    text = re.sub(r'\n', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    # Экранируем кавычки, если нужно
    text = text.replace('"', '""')
    return text

# Создание нового формата
with open('output_filtered_more0_jokes.txt', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        theme = row['theme']
        joke = clean_text(row['text'])
        rating = row['rating']
        
        # Форматируем строку согласно требуемому формату
        formatted_joke = f"<|context|> {theme} <|joke|> {joke}\n"
        f.write(formatted_joke)

print("Файл успешно преобразован в output_jokes.txt")