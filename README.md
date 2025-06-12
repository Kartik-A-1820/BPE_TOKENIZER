# Byte Pair Encoding (BPE) Tokenizer

A minimal and interpretable Byte Pair Encoding (BPE) tokenizer implemented in Python from scratch.

## 📦 Project Structure

```
bpe_tokenizer_project/
|
├── bpe/
│   ├── __init__.py         # Package initializer
│   ├── trainer.py          # Trainer class: learns merge rules and vocabulary
│   ├── encoder.py          # Encoder class: encodes text using merge rules
│   ├── decoder.py          # Decoder class: decodes token IDs to text
|
├── bpe_tokenizer.pkl       # Example trained tokenizer saved with special tokens
├── example.ipynb           # Jupyter notebook to demonstrate usage
└── README.md               # This file
```

## 🚀 Quickstart

```python
from bpe import BPETrainer, BPEEncoder, BPEDecoder

text = "some text sample"
trainer = BPETrainer()
trainer.fit(text, vocab_size=300)
trainer.add_special_tokens(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
trainer.save_pickle("bpe_tokenizer.pkl")

encoder = BPEEncoder.from_pickle("bpe_tokenizer.pkl")
decoder = BPEDecoder.from_pickle("bpe_tokenizer.pkl")

encoded = encoder.encode("<BOS> some text sample <EOS>")
print("Encoded:", encoded)
print("Decoded:", decoder.decode(encoded))
```

## 🧠 How It Works

* Text is tokenized at the byte level (UTF-8) or with optional regex.
* The most frequent byte pairs are merged into new tokens to reduce sequence length.
* Merge rules are learned until a desired vocab size is reached (must be > 256).
* Special tokens can be added (e.g., `<PAD>`, `<BOS>`, `<EOS>`), and they are preserved during encoding/decoding.

## ⚙️ Features

* Fully functional BPE tokenizer with:

  * Merge rule training
  * Custom vocabulary size
  * Pickle/JSON save/load
  * Special token support

## ⚠️ Limitations

* No parallel processing (training is single-threaded).
* Designed for **educational** and **research** purposes — not optimized for production-scale corpora.
* Merge rules apply only at the **byte level** unless regex is provided.

## 📖 References

* [BPE Algorithm - Sennrich et al., 2015](https://aclanthology.org/P16-1162/)
* HuggingFace tokenizers (for advanced production versions)
* See `bpe_implementation.ipynb` for detailed explanation of the algorithm and implementation steps.

---

MIT License © 2025
