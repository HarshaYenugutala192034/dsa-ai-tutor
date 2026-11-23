import json
import re
import time
from pathlib import Path

from gensim.models import Word2Vec

# Training stories from session4
DATA_FILE = Path(__file__).resolve().parent.parent / "session4" / "model" / "data" / "train.jsonl"
MAX_STORIES = 10000


def read_sentences():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for count, line in enumerate(file):
            if count >= MAX_STORIES:
                break
            if line.strip():
                story = json.loads(line)["text"].lower()
                for sentence in re.split(r"[.!?\n]+", story):
                    words = re.findall(r"[a-z']+", sentence)
                    if len(words) > 2:
                        yield words


sentences = list(read_sentences())
print(f"Training sentences: {len(sentences):,}")

start = time.perf_counter()
model = Word2Vec(
    sentences,
    vector_size=50,   # embedding dimension N
    window=3,         # context window
    min_count=5,      # ignore very rare words
    sg=1,             # 1 = Skip-Gram, 0 = CBOW
    epochs=5,
    workers=4,
    seed=42,
)
print(f"Trained in {time.perf_counter() - start:.1f} s, vocabulary = {len(model.wv):,} words")

for word in ["dog", "king", "happy", "mom"]:
    if word in model.wv:
        print(f"\nWords closest to '{word}':")
        for neighbour, score in model.wv.most_similar(word, topn=5):
            print(f"  {neighbour:<12}{score:.3f}")

print("\nCosine similarity")
for a, b in [("cat", "dog"), ("cat", "car"), ("boy", "girl")]:
    if a in model.wv and b in model.wv:
        print(f"  {a}, {b}: {model.wv.similarity(a, b):.3f}")

model.save(str(Path(__file__).resolve().parent / "word2vec_stories.model"))
print("\nModel saved: word2vec_stories.model")
