"""Same text tokenization/window as Exp2; append 2 tokens per actual tree depth."""
from b4lib import require
from hierarchy_adapter import targets

def encode(record, tokenizer, hierarchy, text_max_length=50):
    require(text_max_length == 50, 'Original text window must remain 50')
    tokens = tokenizer(record['text'], truncation=True, max_length=text_max_length,
        add_special_tokens=True, padding=False)['input_ids']
    require(tokens[0] == tokenizer.cls_token_id and tokens[-1] == tokenizer.sep_token_id, 'Wrong tokenizer')
    require(all(0 <= i < tokenizer.vocab_size for i in tokens), 'Text vocabulary changed')
    c, d = hierarchy['num_nodes'], hierarchy['num_depths']
    mask_id = tokenizer.vocab_size + c + d
    prefix = sum(([tokenizer.vocab_size + c + layer, mask_id] for layer in range(d)), [])
    ids = tokens[:-1] + prefix + [tokenizer.sep_token_id]
    total = text_max_length + 2 * d
    require(len(ids) <= total and ids.count(mask_id) == d, 'Invalid depth prompt layout')
    attention = [1] * len(ids) + [0] * (total - len(ids))
    ids += [tokenizer.pad_token_id] * (total - len(ids))
    supervised, fine = targets(record['labels'], hierarchy)
    return {'input_ids': ids, 'attention_mask': attention, 'labels': supervised, 'fine': fine}

def tensors(records, tokenizer, hierarchy, text_max_length):
    import torch
    require(bool(records), 'Empty data')
    encoded = [encode(r, tokenizer, hierarchy, text_max_length) for r in records]
    return {k: torch.tensor([r[k] for r in encoded], dtype=torch.long) for k in encoded[0]}
