"""Small offline word/punctuation vocabulary fitted on selected training text."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


class Tokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
    PATTERN = r"\w+(?:['’]\w+)*|[^\w\s]"

    def __init__(self, vocabulary):
        if vocabulary[:4] != self.SPECIAL or len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Invalid tokenizer vocabulary")
        self.vocabulary = vocabulary
        self.lookup = {token: i for i, token in enumerate(vocabulary)}

    @classmethod
    def tokenize(cls, text):
        return re.findall(cls.PATTERN, text.lower().replace("’", "'"))

    @classmethod
    def fit(cls, captions, max_vocab=4096, min_frequency=1):
        counts = Counter(token for caption in captions for token in cls.tokenize(caption))
        tokens = sorted((token for token, n in counts.items() if n >= min_frequency), key=lambda token: (-counts[token], token))
        return cls(cls.SPECIAL + tokens[:max_vocab - 4])

    def encode(self, text):
        return [self.lookup.get(token, self.UNK) for token in self.tokenize(text)]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == self.EOS:
                break
            if i not in (self.PAD, self.BOS):
                tokens.append(self.vocabulary[i])
        text = " ".join(tokens)
        return re.sub(r"\s+([.,!?;:%])", r"\1", text)

    def to_dict(self):
        return {"version": 1, "kind": "lowercase_word_punctuation", "pattern": self.PATTERN, "vocabulary": self.vocabulary}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != 1 or value.get("kind") != "lowercase_word_punctuation" or value.get("pattern") != cls.PATTERN:
            raise ValueError("Unsupported tokenizer format")
        return cls(value["vocabulary"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.vocabulary)
