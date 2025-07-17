from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("datasets\\v2joke-tokenizer.json")
specials = ["<|pad|>", "<|context|>", "<|joke|>", "<|endoftext|>"]

for tok in specials:
    print(f"{tok}: {tokenizer.token_to_id(tok)}")
