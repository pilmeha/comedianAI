import torch
import torch.nn.functional as F
from v2model import JokeModel
from tokenizers import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_joke(model, tokenizer, context, max_length=50, temperature=1.0, top_k=5):
    model.eval()
    model = model.to(DEVICE)

    input_text = f"<|context|> {context} <|joke|>"
    input_ids = tokenizer.encode(input_text).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    generated = input_tensor

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            sampled_idx = torch.multinomial(top_k_probs[0], 1)
            next_token = top_k_indices[0, sampled_idx]

        # Убедимся, что next_token имеет ту же размерность
        next_token = next_token.unsqueeze(0)  # shape [1]
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
            break

    # Декодируем начиная СРАЗУ после <|joke|>
    decoded = tokenizer.decode(generated[0].tolist())
    print("🧾 Сгенерировано:", repr(decoded))

    # Попробуем найти <|joke|>
    if "<|joke|>" in decoded:
        joke_start = decoded.find("<|joke|>") + len("<|joke|>")
        joke = decoded[joke_start:]
    else:
        joke = decoded  # если модель не вставила тег, берём всё

    return joke.replace("<|endoftext|>", "").strip()


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("datasets\\v2joke-tokenizer.json")
    print(tokenizer.token_to_id("<|joke|>"))
    print(tokenizer.token_to_id("<|endoftext|>"))

    model = JokeModel(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(torch.load("checkpoints/joke_gen_epoch20.pt", map_location=DEVICE))

    context = input("Введите тему шутки: ").strip()
    joke = generate_joke(model, tokenizer, context)
    print(f"\nКонтекст: {context}\nШутка: {joke}")

