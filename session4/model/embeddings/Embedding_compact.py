import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. File paths (run from the project root)
TOKENIZED_FILE = Path("session4/model/tokenization/output/tokenized_data.jsonl")
OUTPUT_DIR = Path("session4/model/embeddings/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 128
SAMPLE_STORIES = 50
torch.manual_seed(42)

# 2. Read tokenized data
all_stories = []
with open(TOKENIZED_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            all_stories.append(json.loads(line))

VOCAB_SIZE = max(max(story["token_ids"]) for story in all_stories) + 1
MAX_SEQ_LEN = max(len(story["token_ids"]) for story in all_stories)
print("Stories:", len(all_stories))
print("Vocabulary size:", VOCAB_SIZE)
print("Maximum sequence length:", MAX_SEQ_LEN)


# 3. Token and Position Embedding class (unchanged)
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


embedding_layer = TokenAndPositionEmbedding(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN)
parameters = sum(p.numel() for p in embedding_layer.parameters())
print("Trainable parameters:", f"{parameters:,}")

# 4. Save the layer itself (a few MB) instead of every story's vectors
layer_file = OUTPUT_DIR / "embedding_layer.pt"
torch.save(
    {
        "state_dict": embedding_layer.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "max_seq_len": MAX_SEQ_LEN,
    },
    layer_file,
)

# 5. Save embeddings for a small sample of stories, for inspection
sample = {}
for i, story in enumerate(all_stories[:SAMPLE_STORIES]):
    ids = torch.tensor(story["token_ids"], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        vectors = embedding_layer(ids).squeeze(0)
    sample[f"story_{i + 1}"] = {
        "tokens": story["tokens"],
        "token_ids": story["token_ids"],
        "final_embeddings": vectors,
    }

sample_file = OUTPUT_DIR / "sample_embeddings.pt"
torch.save(sample, sample_file)

print(f"\nSaved layer:  {layer_file}  ({layer_file.stat().st_size / 1024 ** 2:.1f} MB)")
print(f"Saved sample: {sample_file}  ({sample_file.stat().st_size / 1024 ** 2:.1f} MB)")

# 6. What the embeddings look like (first story)
first = all_stories[0]
id_list = first["token_ids"]
ids = torch.tensor(id_list, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    out = embedding_layer(ids)
    print("\nFirst story:", len(id_list), "tokens -> embedding shape", tuple(out.shape))

    # Untrained token vectors are almost unrelated (cosine close to 0)
    table = embedding_layer.token_embedding.weight
    cosine = F.cosine_similarity(table[id_list[0]], table[id_list[1]], dim=0).item()
    print(f"Cosine of tokens '{first['tokens'][0]}' and '{first['tokens'][1]}': {cosine:.3f}")

    # The same token at two positions: same token vector, different final vector
    repeated = next((t for t, c in Counter(id_list).items() if c > 1), None)
    if repeated is not None:
        p1, p2 = [i for i, t in enumerate(id_list) if t == repeated][:2]
        print(f"Token id {repeated} at positions {p1} and {p2}:")
        print("  token vectors identical:", True)
        print("  final vectors identical:", torch.allclose(out[0, p1], out[0, p2]))
