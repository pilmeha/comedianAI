from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import math

# Загружаем токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'additional_special_tokens': ['<|context|>', '<|joke|>']
    })

model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.gradient_checkpointing_enable()
model.resize_token_embeddings(len(tokenizer))

# Загружаем датасет
def load_dataset_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Разделяем на контекст и шутку (если нужно)
    contexts = []
    jokes = []
    for line in lines:
        if "<|context|>" in line and "<|joke|>" in line:
            parts = line.split("<|joke|>")
            contexts.append(parts[0].replace("<|context|>", "").strip())
            jokes.append(parts[1].strip())
    
    return {"context": contexts, "joke": jokes}

data = load_dataset_from_txt("translated_filtered_more1_jokes.txt")
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
val_data = dataset["test"]

# Токенизация
def tokenize_function(examples):
    # Объединяем контекст и шутку в один текст
    texts = [f"<|context|> {ctx} <|joke|> {joke}" for ctx, joke in zip(examples["context"], examples["joke"])]
    return tokenizer(texts, truncation=True, max_length=96, padding="max_length")

tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["context", "joke"])
tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=["context", "joke"])


def compute_metrics(eval_pred):
    loss = eval_pred.loss
    return {
        "eval_loss": loss,
        "perplexity": math.exp(loss) if loss < 100 else float("inf"),
    }

# Аргументы обучения
training_args = TrainingArguments(
    dataloader_num_workers=2,
    fp16=False,
    output_dir="./joke-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=500,
    report_to="tensorboard",
    logging_dir="./logs",
    optim="adamw_torch",  # более эффективный оптимизатор
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,  
)

if __name__ == '__main__':
    # Запуск обучения
    import torch
    print("💻 Используемое устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    trainer.train()
    torch.cuda.empty_cache()

    # Оценка модели на тестовой выборке
    # trainer.args.device = torch.device("cpu")
    # eval_results = trainer.evaluate()
    # print("Evaluation results:", eval_results)
    model.save_pretrained("trained-joke-distilgpt2_v3")
    tokenizer.save_pretrained("trained-joke-distilgpt2_v3")
