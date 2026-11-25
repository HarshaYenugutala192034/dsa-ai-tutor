import json
from pathlib import Path
from tokenizers import Tokenizer

# Paths (relative to project root)
TRAIN_FILE = Path("session4/model/data/train.jsonl")
TOKENIZER_FILE = Path("session4/model/tokenization/output/bpe_tokenizer.json")
OUTPUT_FILE = Path("session4/model/tokenization/output/tokenized_data.jsonl")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load the trained tokenizer
tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

print("Tokenizer loaded successfully.")
print("Creating tokenized_data.jsonl ...")

count = 0
with open(TRAIN_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue

        story = json.loads(line)["text"]
        if not story.strip():
            continue

        encoding = tokenizer.encode(story)

        record = {
            "text": story,
            "tokens": encoding.tokens,
            "token_ids": encoding.ids
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

        if count % 5000 == 0:
            print(f"  Processed {count} stories...")

print(f"\nDone! Tokenized {count} stories.")
print(f"Saved at: {OUTPUT_FILE.resolve()}")