from transformers.tokenization_utils import PreTrainedTokenizerBase
import numpy as np

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label2id: dict, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        # rebuild text from tokens
        text, labels, token_map = [], [], []

        for idx, (t, l, ws) in enumerate(
            zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])
        ):
            text.append(t)
            labels.extend([l] * len(t))
            token_map.extend([idx]*len(t))

            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

        text = "".join(text)
        labels = np.array(labels)

        # actual tokenization
        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(self.label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map}