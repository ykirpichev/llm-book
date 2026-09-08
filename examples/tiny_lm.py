"""A trainable character bigram model: lifecycle reference, not a transformer."""
import json
import math


class Bigram:
    def __init__(self, vocabulary):
        if not vocabulary or any(not isinstance(c, str) or len(c) != 1 for c in vocabulary):
            raise ValueError("vocabulary must contain characters")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("duplicate vocabulary entry")
        self.vocabulary = tuple(vocabulary)
        self.index = {c: i for i, c in enumerate(self.vocabulary)}
        self.logits = [[0.0] * len(vocabulary) for _ in vocabulary]

    def probabilities(self, character):
        row = self.logits[self.index[character]]
        peak = max(row)
        values = [math.exp(x - peak) for x in row]
        total = sum(values)
        return [x / total for x in values]

    def loss_and_gradient(self, sequences):
        n = len(self.vocabulary)
        counts = [[0] * n for _ in range(n)]
        total = 0
        for sequence in sequences:
            ids = [self.index[c] for c in sequence]
            for a, b in zip(ids, ids[1:]):
                counts[a][b] += 1
                total += 1
        if not total:
            raise ValueError("need at least one next-character target")
        gradient = [[0.0] * n for _ in range(n)]
        loss = 0.0
        for i, row in enumerate(self.logits):
            peak = max(row)
            log_z = peak + math.log(sum(math.exp(x - peak) for x in row))
            mass = sum(counts[i])
            for j in range(n):
                loss += counts[i][j] * (log_z - row[j]) / total
                gradient[i][j] = (mass * math.exp(row[j] - log_z) - counts[i][j]) / total
        return loss, gradient

    def fit(self, sequences, *, steps=200, learning_rate=2.0):
        if not isinstance(steps, int) or steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("invalid training schedule")
        sequences = list(sequences)
        for _ in range(steps):
            _, gradient = self.loss_and_gradient(sequences)
            for i, row in enumerate(self.logits):
                for j in range(len(row)):
                    row[j] -= learning_rate * gradient[i][j]
        return self

    def generate(self, prompt, *, max_new_tokens=6):
        if not prompt or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("need a prompt and a nonnegative generation bound")
        for character in prompt:
            self.index[character]
        text = prompt
        for _ in range(max_new_tokens):
            row = self.logits[self.index[text[-1]]]
            text += self.vocabulary[max(range(len(row)), key=lambda i: row[i])]
        return text

    def to_json(self):
        return json.dumps({"format": 1, "vocabulary": self.vocabulary, "logits": self.logits}, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if data["format"] != 1:
            raise ValueError("unsupported checkpoint format")
        model = cls(data["vocabulary"])
        rows = data["logits"]
        n = len(model.vocabulary)
        if len(rows) != n or any(len(row) != n for row in rows):
            raise ValueError("invalid checkpoint shape")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for row in rows for x in row):
            raise ValueError("invalid checkpoint values")
        model.logits = rows
        return model


if __name__ == "__main__":
    train = ["ababa", "babab"]
    model = Bigram(sorted(set("".join(train))))
    before, _ = model.loss_and_gradient(train)
    model.fit(train)
    after, _ = model.loss_and_gradient(train)
    restored = Bigram.from_json(model.to_json())
    print(f"training loss: {before:.3f} -> {after:.3f}")
    print(restored.generate("a"))
