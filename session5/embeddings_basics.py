import math
from collections import Counter

# ---- 1. One-hot vectors: every pair has dot product 0 (slide 7) ----
vocab = ["The", "cat", "dog", "drinks"]


def one_hot(word):
    return [1 if w == word else 0 for w in vocab]


cat, dog = one_hot("cat"), one_hot("dog")
print("cat =", cat, " dog =", dog)
print("cat . dog =", sum(a * b for a, b in zip(cat, dog)))

# ---- 2. Bag-of-Words: word order is lost (slide 9) ----
s1 = "the dog bites the man".split()
s2 = "the man bites the dog".split()
bow_vocab = ["the", "dog", "bites", "man"]
bow1 = [Counter(s1)[w] for w in bow_vocab]
bow2 = [Counter(s2)[w] for w in bow_vocab]
print("\nBoW 'The dog bites the man':", bow1)
print("BoW 'The man bites the dog':", bow2)
print("Identical vectors:", bow1 == bow2)

# ---- 3. TF-IDF worked example from the lecture (slide 12) ----
N = 100
document_frequency = {"NLP": 10, "Machine": 2, "Learning": 50}
document = ["NLP"] + ["Machine"] * 2 + ["Learning"] * 2 + ["x"] * 5  # 10 words
counts = Counter(document)

print("\nTF-IDF (100 documents, a 10-word document)")
print(f"{'Word':<10}{'TF':>6}{'IDF':>8}{'TF-IDF':>9}")
for word, df in document_frequency.items():
    tf = counts[word] / len(document)
    idf = math.log10(N / df)
    print(f"{word:<10}{tf:>6.2f}{idf:>8.3f}{tf * idf:>9.3f}")

# ---- 4. Cosine similarity of dense vectors (slide 15) ----
vectors = {
    "King": [0.72, 0.15, -0.44, 0.89],
    "Queen": [0.70, 0.17, -0.40, 0.92],
    "Car": [-0.61, 0.82, 0.13, -0.25],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


print("\nCosine similarity")
print("King, Queen:", round(cosine(vectors["King"], vectors["Queen"]), 3))
print("King, Car  :", round(cosine(vectors["King"], vectors["Car"]), 3))
