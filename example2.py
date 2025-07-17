from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Загрузка предобученной модели и токенизатора
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Создаем pipeline для генерации текста
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Тестируем на промпте
prompt = "Судья сказал:"
result = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
print("Стандартная DistilGPT2:\n", result[0]["generated_text"])

from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Укажите путь к вашей обученной модели
custom_model_path = "models\\trained-joke-distilgpt2_v3"  # или "./joke-gpt2"

# Загрузка кастомной модели (должны быть файлы config.json, pytorch_model.bin и т.д.)
tokenizer = GPT2Tokenizer.from_pretrained(custom_model_path)
model = GPT2LMHeadModel.from_pretrained(custom_model_path)

# Проверка специальных токенов
if "<|context|>" not in tokenizer.additional_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|context|>', '<|joke|>']})
    model.resize_token_embeddings(len(tokenizer))

# Генерация с учетом вашего формата
def generate_joke(context):
    input_text = f"<|context|> {context} <|joke|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id)
    
    output = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_length=100,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id = tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=False)

# Тест
print("\nВаша модель:")
print(generate_joke("студент"))