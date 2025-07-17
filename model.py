import torch
import torch.nn as nn
import torch.nn.functional as F


class JokeGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super(JokeGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=False)

        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=False)

        self.attn = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_input, joke_input):
        # context_input: (B, C_len), joke_input: (B, J_len)
        embedded_context = self.embedding(context_input)  # (B, C_len, E)
        encoder_outputs, hidden = self.encoder(embedded_context)  # (B, C_len, H), (num_layers, B, H)

        embedded_joke = self.embedding(joke_input)  # (B, J_len, E)
        decoder_outputs, _ = self.decoder(embedded_joke, hidden)  # (B, J_len, H)

        # Attention mechanism (dot product attention simplified)
        # Calculate attention weights for each decoder time step
        attn_weights = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))  # (B, J_len, C_len)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, J_len, C_len)

        # Context vector
        context_vector = torch.bmm(attn_weights, encoder_outputs)  # (B, J_len, H)

        combined = torch.cat((decoder_outputs, context_vector), dim=-1)  # (B, J_len, 2H)
        combined = self.attn(combined)  # (B, J_len, H)

        logits = self.out(combined)  # (B, J_len, V)
        return logits


if __name__ == '__main__':
    model = JokeGenerator(vocab_size=5000)
    x = torch.randint(0, 5000, (4, 20))  # контексты
    y = torch.randint(0, 5000, (4, 16))  # шутки (вход)
    output = model(x, y)
    print(output.shape)  # (B, J_len, V)
