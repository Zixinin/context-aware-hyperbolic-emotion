import hashlib, json, os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

root=Path(os.environ['FYP_ROOT'])
revision='cd5ef92a9fb2f889e972770a36d4ed042daf221e'
target=root/'assets/pretrained/bert-base-cased'/revision
snapshot_download(
    repo_id='google-bert/bert-base-cased',
    revision=revision,
    local_dir=str(target),
    max_workers=2,
    allow_patterns=[
        'config.json','tokenizer.json','tokenizer_config.json',
        'vocab.txt','model.safetensors','pytorch_model.bin'
    ]
)
assert (target/'model.safetensors').is_file()
assert (target/'pytorch_model.bin').is_file()
tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
config=AutoConfig.from_pretrained(target,local_files_only=True)
assert config.model_type=='bert' and config.hidden_size==768 and config.num_hidden_layers==12
assert tokenizer.do_lower_case is False

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

record={
    'status':'PINNED_RAW_BERT_ASSETS_READY',
    'repository':'google-bert/bert-base-cased',
    'revision':revision,
    'path':str(target),
    'files_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()}
}
(root/'environment/ubuntu/pretrained_assets.json').write_text(
    json.dumps(record,indent=2)+'\n'
)
print('PINNED_RAW_BERT_ASSETS_READY',target,flush=True)
