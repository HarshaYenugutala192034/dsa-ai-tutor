import numpy as np

np.random.seed(42)

# ---- Data: "The cat sat on the mat", window size 1 ----
sentence = ["the", "cat", "sat", "on", "the", "mat"]
vocab = ["the", "cat", "sat", "on", "mat"]          # V = 5
index = {word: i for i, word in enumerate(vocab)}
V, N = len(vocab), 3                                # embedding size N = 3

# CBOW training pairs: (context words) -> centre word
pairs = []
for i, centre in enumerate(sentence):
    context = [sentence[j] for j in (i - 1, i + 1) if 0 <= j < len(sentence)]
    pairs.append(([index[w] for w in context], index[centre]))

print("CBOW training pairs:")
for context, centre in pairs:
    print("  ", [vocab[c] for c in context], "->", vocab[centre])

# ---- Weights: W (V x N) is the embedding table, W2 (N x V) is W' ----
W = np.random.uniform(-0.5, 0.5, (V, N))
W2 = np.random.uniform(-0.5, 0.5, (N, V))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


learning_rate = 0.1
for epoch in range(1, 501):
    total_loss = 0.0
    for context, centre in pairs:
        h = W[context].mean(axis=0)          # average of context embeddings
        p = softmax(h @ W2)                  # output layer + softmax
        total_loss += -np.log(p[centre])     # loss = -log P(centre | context)

        # Backpropagation
        dz = p.copy()
        dz[centre] -= 1
        dW2 = np.outer(h, dz)
        dh = W2 @ dz
        W2 -= learning_rate * dW2
        for c in context:
            W[c] -= learning_rate * dh / len(context)

    if epoch in (1, 100, 250, 500):
        print(f"Epoch {epoch:3d}  loss = {total_loss / len(pairs):.4f}")


# ---- Predict a centre word from its context (slide 18) ----
def predict(context_words):
    ids = [index[w] for w in context_words]
    probs = softmax(W[ids].mean(axis=0) @ W2)
    print(f"\nContext {context_words} -> probabilities")
    for word, prob in sorted(zip(vocab, probs), key=lambda x: -x[1]):
        print(f"  {word:<5}{prob:.3f}")


predict(["cat", "on"])    # unambiguous: the answer is "sat"
predict(["the", "sat"])   # ambiguous: see the note below

print(
    "\nNote: CBOW averages the context vectors, so word order is lost. "
    "('the', 'sat') and ('sat', 'the') give the same average, but one "
    "should predict 'cat' and the other 'on'. The model splits the "
    "probability between them."
)

# ---- The embedding table is the matrix W (slide 26) ----
print("\nLearned embeddings (rows of W):")
for word in vocab:
    print(f"  {word:<5}", np.round(W[index[word]], 3))
