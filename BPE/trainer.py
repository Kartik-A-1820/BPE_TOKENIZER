# --- trainer.py ---
import json
import pickle
import regex as re

class BPETrainer:
    """
    Trainer for learning Byte Pair Encoding (BPE) merge rules and vocabulary from raw text.
    """

    def __init__(self):
        """
        Initializes internal storage for merge rules and vocabulary.
        """
        self._merge_dict = {}
        self._vocab = {}
        self.vocab_size = None
        self.special_tokens = {}

    def _get_stats(self, tokens):
        """
        Counts frequencies of adjacent token pairs.

        Args:
            tokens (List[int]): List of token IDs.

        Returns:
            dict: Mapping of (token1, token2) -> frequency.
        """
        freq = {}
        for i in range(len(tokens) - 1):
            pair = tokens[i], tokens[i + 1]
            freq[pair] = freq.get(pair, 0) + 1
        return freq

    def _merge(self, tokens, pair, idx):
        """
        Replaces occurrences of a given pair in the token sequence with a new token ID.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _get_vocab(self):
        """
        Constructs the vocabulary from current merge rules.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), x in self._merge_dict.items():
            vocab[x] = vocab[p0] + vocab[p1]
        return vocab

    def fit(self, text, vocab_size=384, regex=None):
        """
        Learns BPE merge rules and constructs a vocabulary from input text.
        """
        if vocab_size <= 256:
            return None  # Invalid size

        self.vocab_size = vocab_size

        # --- Step 1: Tokenization ---
        if regex:
            tokens = []
            last_idx = 0
            for match in re.finditer(regex, text):
                start, end = match.span()
                if start > last_idx:
                    unmatched = text[last_idx:start]
                    tokens.extend(unmatched.encode("utf-8"))
                tokens.extend(match.group().encode("utf-8"))
                last_idx = end
            if last_idx < len(text):
                tokens.extend(text[last_idx:].encode("utf-8"))
        else:
            tokens = list(text.encode("utf-8"))

        if vocab_size > len(tokens):
            return None  # Not enough data for meaningful merges

        # --- Step 2: Merge pairs ---
        idx = 256
        for _ in range(vocab_size - 256):
            stats = self._get_stats(tokens)
            if not stats:
                break
            freq_pair = max(sorted([(v, k) for k, v in stats.items()], reverse=True))[1]
            self._merge_dict[freq_pair] = idx
            tokens = self._merge(tokens, freq_pair, idx)
            idx += 1

        # --- Step 3: Final Vocab ---
        self._vocab = self._get_vocab()
        return self._merge_dict, self._vocab

    def add_special_tokens(self, special_tokens):
        """
        Adds special tokens like <PAD>, <UNK>, etc. to the vocabulary.

        Args:
            special_tokens (List[str])
        """
        reverse_vocab = {v: k for k, v in self._vocab.items()}
        current_max_id = max(self._vocab.keys(), default=255)

        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in reverse_vocab:
                token_id = reverse_vocab[token_bytes]
            else:
                current_max_id += 1
                token_id = current_max_id
                self._vocab[token_id] = token_bytes
            self.special_tokens[token] = token_id

    def get_special_tokens(self):
        return self.special_tokens

    def get_merge_dict(self):
        return self._merge_dict

    def get_vocab(self):
        return self._vocab

    def save_json(self, path="bpe_tokenizer.json"):
        """
        Saves merge rules and vocab to a JSON file.
        """
        with open(path, "w") as f:
            json.dump({
                "merge_dict": {str(k): v for k, v in self._merge_dict.items()},
                "vocab": {str(k): v.decode("utf-8", errors="replace") for k, v in self._vocab.items()},
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens
            }, f)

    def load_json(self, path="bpe_tokenizer.json"):
        """
        Loads merge rules and vocab from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)
            self._merge_dict = {eval(k): v for k, v in data["merge_dict"].items()}
            self._vocab = {int(k): v.encode("utf-8", errors="replace") for k, v in data["vocab"].items()}
            self.vocab_size = data["vocab_size"]
            self.special_tokens = data.get("special_tokens", {})

    def save_pickle(self, path="bpe_tokenizer.pkl"):
        """
        Saves merge rules and vocab as a binary pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump({
                "merge_dict": self._merge_dict,
                "vocab": self._vocab,
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens
            }, f)

    def load_pickle(self, path="bpe_tokenizer.pkl"):
        """
        Loads merge rules and vocab from a pickle file.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._merge_dict = data["merge_dict"]
            self._vocab = data["vocab"]
            self.vocab_size = data["vocab_size"]
            self.special_tokens = data.get("special_tokens", {})
