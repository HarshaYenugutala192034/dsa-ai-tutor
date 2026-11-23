
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter
import sentencepiece as spm
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "compare_output"
OUT_DIR.mkdir(exist_ok=True)
CORPUS_FILE = OUT_DIR / "corpus.txt"

#Create Multilingual DSA Corpus

corpus = {
    "English": [
        "Data structures organize data efficiently",
        "Arrays store elements in contiguous memory",
        "Linked lists consist of connected nodes",
        "Stacks follow the last in first out principle",
        "Queues follow the first in first out principle"
    ],

    "Telugu": [
        "డేటా స్ట్రక్చర్స్ అనేవి డేటాను క్రమబద్ధంగా నిల్వ చేస్తాయి",
        "అర్రే అనేది ఒకే రకమైన డేటాను నిల్వ చేస్తుంది",
        "లింక్డ్ లిస్ట్ నోడ్స్ తో నిర్మించబడుతుంది",
        "స్టాక్ లాస్ట్ ఇన్ ఫస్ట్ అవుట్ సూత్రాన్ని అనుసరిస్తుంది",
        "క్యూ ఫస్ట్ ఇన్ ఫస్ట్ అవుట్ సూత్రాన్ని అనుసరిస్తుంది"
    ],

    "Hindi": [
        "डेटा स्ट्रक्चर डेटा को व्यवस्थित रूप से रखता है",
        "ऐरे समान प्रकार के तत्वों को संग्रहीत करता है",
        "लिंक्ड लिस्ट नोड्स से बनी होती है",
        "स्टैक लास्ट इन फर्स्ट आउट सिद्धांत का पालन करता है",
        "क्यू फर्स्ट इन फर्स्ट आउट सिद्धांत का पालन करती है"
    ],

    "Mixed": [
        "Data structures చాలా important",
        "Array లో elements store అవుతాయి",
        "Linked list లో nodes ఉంటాయి",
        "Stack follows LIFO principle",
        "Queue లో FIFO principle ఉపయోగిస్తారు"
    ]
}

# Combine all sentences for tokenizer training
all_sentences = []

for sentences in corpus.values():
    all_sentences.extend(sentences)

# Save combined corpus
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for line in all_sentences:
        f.write(line + "\n")

print("Total training sentences:", len(all_sentences))

# Train BPE Tokenizer

bpe = Tokenizer(BPE(unk_token="[UNK]"))
bpe.train(
    [str(CORPUS_FILE)],
    BpeTrainer(
        vocab_size=200,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
)

# Train WordPiece Tokenizer

wp = Tokenizer(WordPiece(unk_token="[UNK]"))
wp.pre_tokenizer = Whitespace()

wp.train(
    [str(CORPUS_FILE)],
    WordPieceTrainer(
        vocab_size=200,
        special_tokens=["[UNK]"]
    )
)

# Step 4: Train SentencePiece Unigram Tokenizer

spm.SentencePieceTrainer.train(
    input=str(CORPUS_FILE),
    model_prefix=str(OUT_DIR / "sp"),
    vocab_size=100,
    model_type="unigram",
    minloglevel=2
)

sp = spm.SentencePieceProcessor()
sp.load(str(OUT_DIR / "sp.model"))

# Step 5: Evaluate Tokenizers on Each Language

results = {
    "BPE": {},
    "WordPiece": {},
    "SentencePiece": {}
}

tokenizers = {
    "BPE": bpe,
    "WordPiece": wp,
    "SentencePiece": sp
}

# Process one tokenizer at a time
for tokenizer_name, tokenizer in tokenizers.items():

    print("\nTOKENIZER:", tokenizer_name)
    # Process each language separately
    for language, sentences in corpus.items():

        all_tokens = []

        for sentence in sentences:

            if tokenizer_name == "SentencePiece":
                tokens = tokenizer.encode_as_pieces(sentence)
            else:
                tokens = tokenizer.encode(sentence).tokens

            all_tokens.extend(tokens)

        # Calculate total token count
        total_tokens = len(all_tokens)

        # Store results
        results[tokenizer_name][language] = {
            "first_5_tokens": all_tokens[:5],
            "total_tokens": total_tokens
        }

        # Display language-wise results
        print("\nLanguage:", language)
        print("First 5 Tokens:", all_tokens[:5])
        print("Total Token Count:", total_tokens)

    # Calculate and display vocabulary size only once
    if tokenizer_name == "SentencePiece":
        vocab_size = tokenizer.get_piece_size()
    else:
        vocab_size = len(tokenizer.get_vocab())

    print("\nTotal Vocabulary Size:", vocab_size)

    # Store vocabulary size
    results[tokenizer_name]["vocabulary_size"] = vocab_size


# Summary table: total tokens for the same sentences (fewer = more efficient)
names = ["BPE", "WordPiece", "SentencePiece"]

print("\n========== SUMMARY: total tokens per language ==========")
print(f"{'Language':<10}{'Chars':>7}{'BPE':>7}{'WordPiece':>11}{'SentencePiece':>15}")

for language, sentences in corpus.items():
    chars = sum(len(sentence) for sentence in sentences)
    counts = [results[name][language]["total_tokens"] for name in names]
    print(f"{language:<10}{chars:>7}{counts[0]:>7}{counts[1]:>11}{counts[2]:>15}")

vocab_sizes = [results[name]["vocabulary_size"] for name in names]
print(f"{'Vocab':<10}{'':>7}{vocab_sizes[0]:>7}{vocab_sizes[1]:>11}{vocab_sizes[2]:>15}")
