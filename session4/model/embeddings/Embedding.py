import json
from pathlib import Path
import torch
import torch.nn as nn


# 1. File paths (relative to project root)
TOKENIZED_FILE = Path("session4/model/tokenization/output/tokenized_data.jsonl")
OUTPUT_DIR = Path("session4/model/embeddings/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Embedding configuration
D_MODEL = 128
torch.manual_seed(42)


# 3. Read tokenized data
all_stories = []

with open(TOKENIZED_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            all_stories.append(json.loads(line))


# 4. Find vocabulary size and maximum sequence length
VOCAB_SIZE = max(
    max(story["token_ids"]) for story in all_stories
) + 1

MAX_SEQ_LEN = max(
    len(story["token_ids"]) for story in all_stories
)

print("Vocabulary size:", VOCAB_SIZE)
print("Maximum sequence length:", MAX_SEQ_LEN)


# 5. Token and Position Embedding class
class TokenAndPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError("Sequence exceeds maximum sequence length")

        token_vectors = self.token_embedding(input_ids)

        positions = torch.arange(seq_len, device=input_ids.device)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors.unsqueeze(0)


# 6. Create embedding layer
embedding_layer = TokenAndPositionEmbedding(
    VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN
)


# 7. Generate embeddings for all stories
all_embeddings = {}

for i, story in enumerate(all_stories):

    input_ids = torch.tensor(
        story["token_ids"], dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        embeddings = embedding_layer(input_ids)

    all_embeddings[f"story_{i+1}"] = {
        "tokens": story["tokens"],
        "token_ids": story["token_ids"],
        "final_embeddings": embeddings.squeeze(0)
    }

    if (i + 1) % 5000 == 0:
        print(f"  Embedded {i+1} stories...")


# 8. Save all embeddings
OUTPUT_FILE = OUTPUT_DIR / "all_embeddings.pt"
torch.save(all_embeddings, OUTPUT_FILE)

print("\nEmbedding generation completed.")
print("Total stories:", len(all_embeddings))
print("Embedding dimension:", D_MODEL)
print("Saved at:", OUTPUT_FILE.resolve())