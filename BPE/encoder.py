import pickle

class BPEEncoder:
    def __init__(self, merge_dict, special_tokens=None):
        self.merge_dict = merge_dict
        self.encoded_tokens = []
        self.special_tokens = special_tokens or {}

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

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["merge_dict"], data.get("special_tokens", {}))

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted(self.special_tokens, key=len, reverse=True):
                if text[i:].startswith(token):
                    tokens.append(self.special_tokens[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                tokens.extend(text[i].encode("utf-8"))
                i += 1

        # Apply merge rules
        j = 0
        while j < len(tokens) - 1:
            pair = tokens[j], tokens[j + 1]
            if pair in self.merge_dict:
                tokens = self._merge(tokens, pair, self.merge_dict[pair])
                j = 0
            else:
                j += 1

        self.encoded_tokens = tokens
        return tokens
