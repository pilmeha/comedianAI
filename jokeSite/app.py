from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# === Инициализация ===
MODEL_PATH = "model/trained-joke-distilgpt2_testV4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# === FastAPI ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Главная страница ===
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "joke": None})

# === Генерация шутки ===
@app.post("/", response_class=HTMLResponse)
async def generate(request: Request, context: str = Form(...)):
    input_text = f"<|context|> {context} <|joke|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # attention_mask = (input_ids != tokenizer.pad_token_id)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            # attention_mask,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
        )
    joke = tokenizer.decode(output[0], skip_special_tokens=True).split("<|joke|>")[-1].strip()

    return templates.TemplateResponse("index.html", {"request": request, "joke": joke, "context": context})
