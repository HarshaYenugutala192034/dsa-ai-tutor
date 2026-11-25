
import json
from pathlib import Path
import torch
import torch.nn as nn
from tokenizers import Tokenizer

# 1. File paths
BASE_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = BASE_DIR / "TOKENIZATION" / "output" / "bpe_tokenizer.json"
TRAIN_FILE = BASE_DIR / "DATA" / "train.jsonl"

OUTPUT_DIR = BASE_DIR / "EMBEDDINGS" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Embedding configuration
D_MODEL = 128
torch.manual_seed(42)

# 3. Load trained tokenizer
tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

VOCAB_SIZE = tokenizer.get_vocab_size()

print("Vocabulary size:", VOCAB_SIZE)

# 4. Read stories and tokenize them
all_stories = []

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            story = json.loads(line)["text"]

            encoding = tokenizer.encode(story)

            all_stories.append({
                "tokens": encoding.tokens,
                "token_ids": encoding.ids
            })

# 5. Find maximum sequence length
MAX_SEQ_LEN = max(
    len(story["token_ids"]) for story in all_stories
)

print("Total stories:", len(all_stories))
print("Maximum sequence length:", MAX_SEQ_LEN)

# 6. Token and Position Embedding class
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

# 7. Create embedding layer
embedding_layer = TokenAndPositionEmbedding(
    VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN
)

# 8. Generate embeddings
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

# 9. Save embeddings
OUTPUT_FILE = OUTPUT_DIR / "all_embeddings.pt"
torch.save(all_embeddings, OUTPUT_FILE)

print("\nEmbedding generation completed.")
print("Embedding dimension:", D_MODEL)
print("Saved at:", OUTPUT_FILE.resolve())
