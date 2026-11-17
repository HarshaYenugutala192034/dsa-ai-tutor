import tiktoken

TEXTS = {
    "English": "Data structures and algorithms are important.",
    "Telugu": "డేటా స్ట్రక్చర్స్ మరియు అల్గారిథమ్స్ ముఖ్యమైనవి.",
    "Hindi": "डेटा स्ट्रक्चर और एल्गोरिदम महत्वपूर्ण हैं।",
}

ENCODINGS = ["cl100k_base", "o200k_base"]

# 1. Compare token counts across two vocabularies
for name in ENCODINGS:
    enc = tiktoken.get_encoding(name)
    print(f"\n===== {name} (vocabulary size: {enc.n_vocab:,}) =====")
    print(f"{'Language':<10}{'Characters':>12}{'Tokens':>9}{'Chars/token':>13}")

    for language, text in TEXTS.items():
        ids = enc.encode(text)
        print(
            f"{language:<10}{len(text):>12}{len(ids):>9}"
            f"{len(text) / len(ids):>13.2f}"
        )

# 2. Look inside the tokens: IDs and the text piece for each ID
enc = tiktoken.get_encoding("o200k_base")

for language, text in TEXTS.items():
    ids = enc.encode(text)
    pieces = [
        enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
        for i in ids
    ]
    print(f"\n--- {language} ---")
    print("Token IDs:", ids)
    print("Pieces:   ", pieces)

print(
    "\nNote: a piece shown as the replacement character means the token holds "
    "only part of a character's bytes."
)