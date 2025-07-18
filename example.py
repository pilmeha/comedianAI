from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os


# Загружаем обученную модель и токенизатор
model_path = "models\\trained-joke-distilgpt2_testV4"  # путь к сохраненной модели
# model_path = "distilgpt2"  # путь к сохраненной модели

# Проверка существования пути
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found at: {os.path.abspath(model_path)}")

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Функция для генерации шутки
def generate_joke(context: str) -> str:
    # Подготавливаем входной текст
    input_text = f"<|context|> {context} <|joke|>"
    
    # Токенизируем
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id)
    
    # Генерируем текст
    # with torch.no_grad():
    output = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_length=128,  # Максимальная длина выходного текста
        num_return_sequences=1,  # Количество вариантов
        do_sample=True,  # Включаем случайную генерацию
        top_k=50,  # Ограничиваем выбор топ-50 слов
        top_p=0.9,  # Nucleus sampling
        temperature=0.7,  # "Творческость" модели
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id = tokenizer.eos_token_id,
        # repetition_penalty=1.2, 
        # no_repeat_ngram_size=2,
    )
    
    # Декодируем и возвращаем результат
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Извлекаем только шутку (все что после <|joke|>)
    joke_start = generated_text.find("<|joke|>") + len("<|joke|>")
    joke = generated_text[joke_start:].strip()
    
    # Удаляем лишние токены (если есть)
    joke = joke.replace("<|endoftext|>", "").strip()
    
    return joke

if __name__=='__main__':
# Пример использования
    context = input("Введите тему шутки: ").strip()
    generated_joke = generate_joke(context)
    print(f"Контекст: {context}\nШутка: {generated_joke}")

    contexts = ["кофе", "работа", "кошки", "аниме", "ИИ"]
    for c in contexts:
        print(f"{c}: {generate_joke(c)}")