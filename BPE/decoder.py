import pickle

class BPEDecoder:
    """
    Decodes BPE token sequences back to readable UTF-8 text.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_pickle(cls, path):
        """
        Loads vocab from a saved tokenizer pickle file.

        Args:
            path (str): Path to `.pkl` file saved by BPETrainer.

        Returns:
            BPEDecoder: Ready-to-use decoder instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["vocab"])

    def decode(self, tokens):
        """
        Converts token IDs to UTF-8 string.

        Args:
            tokens (list): Encoded BPE token sequence.

        Returns:
            str: Decoded UTF-8 string.
        """
        try:
            return b"".join(self.vocab.get(x, b"") for x in tokens).decode("utf-8", errors='replace')
        except:
            return "".join(self.vocab.get(str(x), "") for x in tokens)
