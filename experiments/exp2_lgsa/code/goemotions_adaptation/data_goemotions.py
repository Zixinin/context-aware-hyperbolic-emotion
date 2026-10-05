"""Keep original text/IDs/splits; append LGSA prompts after baseline text window."""
import json
import random
from common import require
from hierarchy_goemotions import targets


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sample(records, count, seed):
    indices = sorted(random.Random(seed).sample(range(len(records)), count))
    return [records[i] for i in indices]


def encode(record, tokenizer, hierarchy, text_max_length=50):
    tokens = tokenizer(record['text'], truncation=True, max_length=text_max_length,
                       add_special_tokens=True, padding=False)['input_ids']
    require(tokens[0] == tokenizer.cls_token_id and tokens[-1] == tokenizer.sep_token_id, 'Expected BERT tokenizer')
    require(all(0 <= t < tokenizer.vocab_size for t in tokens), 'Unexpected vocabulary extension')
    # Two depth tokens, two task MASK tokens, final SEP. Original text budget is unchanged.
    prefix = [tokenizer.vocab_size + 32, tokenizer.vocab_size + 34,
              tokenizer.vocab_size + 33, tokenizer.vocab_size + 34, tokenizer.sep_token_id]
    ids = tokens[:-1] + prefix
    total_length = text_max_length + 4
    attention = [1] * len(ids) + [0] * (total_length - len(ids))
    ids += [tokenizer.pad_token_id] * (total_length - len(ids))
    supervised, fine = targets(record['labels'], hierarchy)
    require(len(ids) == total_length and ids.count(tokenizer.vocab_size + 34) == 2, 'Invalid prompt layout')
    return {'input_ids': ids, 'attention_mask': attention, 'labels': supervised, 'fine': fine}


def tensors(records, tokenizer, hierarchy, text_max_length):
    import torch
    encoded = [encode(r, tokenizer, hierarchy, text_max_length) for r in records]
    return {key: torch.tensor([r[key] for r in encoded], dtype=torch.long) for key in encoded[0]}


class TensorRows:
    def __init__(self, data, order=None):
        self.data = data
        self.order = list(range(len(data['input_ids']))) if order is None else order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return {key: value[self.order[index]] for key, value in self.data.items()}
