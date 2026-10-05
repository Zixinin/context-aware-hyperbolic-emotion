"""HPC CPU preflight with a tiny RANDOM BERT, never actual pretrained training."""
import tempfile
from pathlib import Path
import sys
import torch
from common import ROOT, PACKAGE, config, require
from hierarchy_goemotions import load_hierarchy
from data_goemotions import tensors
from geometry import initialize_and_check


def main():
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    sys.path.insert(0, str(ROOT / 'working_lgsa'))
    from models.prompt import Prompt
    from models.loss import HierarchicalGofLoss
    torch.set_num_threads(1); torch.manual_seed(7)
    hierarchy = load_hierarchy(PACKAGE / 'hierarchy.json')
    with tempfile.TemporaryDirectory(prefix='lgsa-tiny-cpu-') as directory:
        temp = Path(directory)
        vocab = ['[PAD]'] + [f'[unused{i}]' for i in range(1,100)] + ['[UNK]','[CLS]','[SEP]','[MASK]']
        vocab += list(hierarchy['labels'].values()) + ['Example', 'text', 'and', 'happy']
        (temp/'vocab.txt').write_text('\n'.join(vocab)+'\n')
        tokenizer = BertTokenizerFast(vocab_file=str(temp/'vocab.txt'), do_lower_case=False)
        tokenizer.save_pretrained(temp)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=128)
        BertForMaskedLM(cfg).save_pretrained(temp)
        data=temp/'model_data'; data.mkdir()
        torch.save(hierarchy['labels'],data/'value_dict.pt')
        model, loading = Prompt.from_pretrained(temp,num_labels=32,path_list=hierarchy['edges'],
            layer=1,graph_type='GAT',data_path=str(data),depth2label=hierarchy['depths'],
            output_loading_info=True,local_files_only=True,attn_implementation='eager')
        require(not any(n.startswith('bert.') for n in loading['missing_keys']), 'Tiny backbone weights missing')
        initialize_and_check(model,tokenizer,hierarchy,ROOT/'working_lgsa',config(),temp)
        records=[{'id':str(i),'text':'Example happy text and '*8,'labels':[17,19] if i%2 else [0,27]} for i in range(4)]
        batch=tensors(records,tokenizer,hierarchy,50);batch.pop('fine')
        model.train()
        output, mlm, classification=model(**batch)
        gof=HierarchicalGofLoss(hierarchy['depths'],hierarchy['parents'])
        loss=output.loss+gof(model.get_input_embeddings().weight()[-35:-3])
        require(bool(torch.isfinite(loss)), 'Tiny pipeline loss is nonfinite')
        loss.backward()
        require(any('.graph.' in n and p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters()),'No tiny GAT gradient')
        model.eval()
        with torch.no_grad():
            _, logits=model.generate(batch['input_ids'],depth2label=hierarchy['depths'])
        require(tuple(logits.shape)==(4,2,32) and bool(torch.isfinite(logits).all()),'Wrong generated logits')
        print('TINY_RANDOM_CPU_PIPELINE_PASS (not pretrained BERT/GPU smoke)', flush=True)


if __name__=='__main__':
    main()
