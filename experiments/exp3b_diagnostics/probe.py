"""Fixed-checkpoint forward/autograd probes; no optimizer exists in this module."""
from collections import Counter, defaultdict
import json
import math
import random
import re
import numpy as np
from analysis import csvwrite, jsonwrite, require, gradient_summary, spearman


def seed_batch(batch_index):
    import torch
    value = 420000 + batch_index
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def term_values(loss_fn, anchor, keys, scores, gold):
    """Call the same official excerpt-derived kernel on the SAME selected terms."""
    import torch
    from upstream.dpt_positive import positive_label_contrastive
    terms = loss_fn.select(scores.detach(), gold.detach())
    result = []
    with torch.no_grad():
        for sample, true, negatives in terms:
            require(len(negatives) == 1, 'This diagnostic requires frozen K=1')
            loss, pos, neg = positive_label_contrastive(anchor[sample].detach(), keys[true:true+1].detach(),
                                                       keys[negatives].detach(), loss_fn.temperature)
            require(bool(torch.isfinite(loss)), 'Nonfinite term loss')
            result.append({'sample_index': sample, 'true_id': true, 'negative_id': negatives[0],
                           'raw_loss': float(loss), 'cosine_margin': float(pos[0]-neg[0])})
    return result


def aggregate_terms(terms, labels, frequencies, batches, weight):
    def table(kind, keys):
        result = []
        raw_total = sum(r['raw_loss'] for r in terms)
        weighted_total = sum(r['weighted_batch_contribution'] for r in terms)
        require(raw_total > 0 and weighted_total > 0, 'No actual peer contributions')
        for key in keys:
            selected = [r for r in terms if r[kind] == key]
            n = len(selected)
            raw = sum(r['raw_loss'] for r in selected)
            weighted = sum(r['weighted_batch_contribution'] for r in selected)
            row = {'label': key, 'eligible_terms': n, 'term_count_share': n/len(terms),
                   'raw_loss_sum': raw, 'raw_loss_mean': raw/n if n else None,
                   'raw_loss_share': raw/raw_total,
                   'weighted_batch_loss_sum': weighted,
                   'weighted_batch_loss_mean_across_all_batches': weighted/batches,
                   'weighted_batch_loss_share': weighted/weighted_total,
                   'scope': 'BEST_EPOCH3_FULL_TRAIN_FIXED_WEIGHT_SNAPSHOT_NOT_HISTORICAL'}
            if kind == 'true_label':
                label_id = next(i for i, r in enumerate(labels) if r['fine_label'] == key)
                row['train_frequency'] = int(frequencies[label_id])
                row['negative_selection_counts'] = json.dumps(dict(Counter(r['negative_label'] for r in selected)), sort_keys=True)
            result.append(row)
        return sorted(result, key=lambda r: -r['raw_loss_sum'])
    names = [r['fine_label'] for r in labels]
    true_rows = table('true_label', names)
    parent_rows = table('parent', ['Positive', 'Negative', 'Ambiguous'])
    negative_rows = table('negative_label', names)
    # Neutral must not appear in any peer term; singleton is not an eligible negative.
    require(all(r['parent'] != 'Neutral' for r in terms), 'Neutral unexpectedly selected')
    pair_rows = []
    for true, neg in sorted({(r['true_label'], r['negative_label']) for r in terms}):
        part = [r for r in terms if r['true_label'] == true and r['negative_label'] == neg]
        pair_rows.append({'true_label': true, 'negative_label': neg, 'selected_count': len(part),
                          'raw_loss_sum': sum(r['raw_loss'] for r in part),
                          'mean_associated_loss': float(np.mean([r['raw_loss'] for r in part])),
                          'weighted_batch_loss_sum': sum(r['weighted_batch_contribution'] for r in part)})
    # Reconcile BOTH raw and 1/N-batch-weighted actual contributions independently.
    for rows in (true_rows, parent_rows, negative_rows):
        require(math.isclose(sum(r['raw_loss_sum'] for r in rows), sum(r['raw_loss'] for r in terms), rel_tol=1e-10),
                'Raw loss contribution reconciliation failed')
        require(math.isclose(sum(r['weighted_batch_loss_share'] for r in rows), 1., abs_tol=1e-10),
                'Weighted contribution shares do not sum to one')
    targeted = [r for r in true_rows if r['eligible_terms'] > 0]
    correlations = {key: spearman([r['train_frequency'] for r in targeted], [r[key] for r in targeted])
                    for key in ('raw_loss_sum', 'raw_loss_mean', 'weighted_batch_loss_sum')}
    summary = {'checkpoint': 'checkpoint_best.pt', 'epoch': 3, 'fixed_seed': 42,
               'batch_rng_seed_rule': '420000 + zero-based batch index', 'batches': batches,
               'terms': len(terms), 'peer_lambda': weight,
               'raw_loss_sum': sum(r['raw_loss'] for r in terms),
               'weighted_mean_peer_loss': sum(r['weighted_batch_contribution'] for r in terms)/batches,
               'top5_true_loss_share': sum(r['raw_loss_share'] for r in true_rows[:5]),
               'top5_negative_loss_share': sum(r['raw_loss_share'] for r in negative_rows[:5]),
               'top10_negative_loss_share': sum(r['raw_loss_share'] for r in negative_rows[:10]),
               'frequency_correlations': correlations,
               'frequency_correlation_scope': 'Labels with eligible peer terms; exclude neutral singleton.',
               'frequency_correlations_all28': {key: spearman([r['train_frequency'] for r in true_rows],
                   [r[key] for r in true_rows]) for key in ('raw_loss_sum', 'raw_loss_mean', 'weighted_batch_loss_sum')},
               'scope': 'All existing training examples once at fixed best weights, train-mode dropout/MLM, no_grad.',
               'limits': 'Not historical accumulated training losses, optimizer pressure, parameter changes, or causal evidence.'}
    return true_rows, parent_rows, negative_rows, pair_rows, summary


def gradient_scope(name):
    if name.startswith('bert.encoder.'):
        return 'BERT_encoder'
    if '.graph.' in name:
        return 'GAT_path'
    if name.startswith('cls.predictions.transform.'):
        return 'shared_MLM_anchor_transform'
    return 'other_shared'


def compare_gradients(named, original, peer, batch_index, weight):
    """Streaming FP64 dot products, avoiding concatenated model-sized vectors."""
    import torch
    aggregates = defaultdict(lambda: [0., 0., 0., 0])
    excluded, names = [], []
    for (name, parameter), a, b in zip(named, original, peer):
        require(parameter.requires_grad, 'Frozen parameter passed to gradient diagnostic')
        if a is None or b is None:
            excluded.append({'parameter': name, 'original_connected': a is not None, 'peer_connected': b is not None})
            continue
        require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'Nonfinite gradient: ' + name)
        names.append(name)
        # Accumulate blocks to bound extra FP64 memory for large vocabulary tables.
        aa = bb = ab = 0.
        af, bf = a.reshape(-1), b.reshape(-1)
        for start in range(0, a.numel(), 1024*1024):
            x, y = af[start:start+1024*1024].double(), bf[start:start+1024*1024].double()
            aa += float((x*x).sum()); bb += float((y*y).sum()); ab += float((x*y).sum())
        scopes = ['full_shared', gradient_scope(name)]
        layer = re.search(r'^bert\.encoder\.layer\.(\d+)\.', name)
        if layer:
            scopes.append('BERT_layer_'+layer[1])
        for scope in scopes:
            value = aggregates[scope]
            value[0] += aa; value[1] += bb; value[2] += ab; value[3] += 1
    rows = []
    for scope in ['full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform'] + \
                 sorted(s for s in aggregates if s.startswith('BERT_layer_')):
        aa, bb, ab, n = aggregates[scope]
        norm_a, norm_b = math.sqrt(aa), math.sqrt(bb)
        cosine = ab/(norm_a*norm_b) if norm_a > 1e-12 and norm_b > 1e-12 else None
        if cosine is not None:
            require(abs(cosine) <= 1+1e-8, 'Bad gradient cosine')
            cosine = max(-1., min(1., cosine))
        rows.append({'batch_index': batch_index, 'scope': scope, 'shared_parameter_tensors': n,
                     'original_gradient_norm': norm_a, 'peer_gradient_norm_unweighted': norm_b,
                     'peer_gradient_norm_weighted': weight*norm_b, 'cosine': cosine,
                     'weighted_peer_to_original_norm_ratio': weight*norm_b/norm_a if norm_a > 1e-12 else None})
    return rows, excluded, names


def run_probes(repo, lgsa, run, cfg, hierarchy, labels, records, out, frequencies, state_digest):
    import torch
    from transformers import AutoTokenizer
    from models.prompt import Prompt
    from models.loss import HierarchicalGofLoss
    from data_goemotions import tensors
    from lgsa_peer import LGSAWithPeer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one available GPU')
    require('4090' in torch.cuda.get_device_name(0) and torch.cuda.mem_get_info()[0] >= 8*1024**3,
            'Expected an idle RTX4090 with at least 8GiB free')
    raw = repo / 'assets/pretrained/bert-base-cased/cd5ef92a9fb2f889e972770a36d4ed042daf221e'
    tokenizer = AutoTokenizer.from_pretrained(str(raw), local_files_only=True, clean_up_tokenization_spaces=False)
    checkpoint = torch.load(run / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    require(checkpoint['epoch'] == 3 and checkpoint['step'] == 8142 and checkpoint['lineage']['run_directory'] == str(run),
            'Unexpected best checkpoint identity')
    initial = torch.load(run / 'initial_geometry.pt', map_location='cpu', weights_only=False)['hgs'][:28]
    key_name = 'bert.embeddings.word_embeddings.new_embedding.weight'
    require(torch.equal(checkpoint['model'][key_name][1:29], initial), 'Best keys differ from frozen HGS')
    model = Prompt.from_pretrained(str(raw), num_labels=32, path_list=hierarchy['edges'], layer=cfg['graph_layers'],
        graph_type=cfg['graph'], data_path=str(run / 'model_data'), depth2label=hierarchy['depths'],
        local_files_only=True, attn_implementation='eager')
    model.init_embedding(); model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    final = torch.load(run / 'checkpoint_resume.pt', map_location='cpu', weights_only=False)
    require(final['epoch'] == 4 and final['step'] == 10856 and final['lineage']['run_directory'] == str(run),
            'Unexpected final checkpoint identity')
    require(torch.equal(final['model'][key_name][1:29], initial), 'Final keys differ from frozen HGS')
    del final
    require(not model.get_input_embeddings().new_embedding.weight.requires_grad, 'Raw keys/prompts unexpectedly trainable')
    model.cuda(); model.train()
    before = state_digest(model)
    adapter = LGSAWithPeer(model, hierarchy, cfg).attach()
    capture = {}
    original_kernel = adapter.loss_fn.forward
    def wrapped(anchor, keys, scores, gold):
        loss, stats = original_kernel(anchor, keys, scores, gold)
        terms = term_values(adapter.loss_fn, anchor, keys, scores, gold)
        require(len(terms) == stats['number_of_peer_terms'], 'Actual term count differs from native peer kernel')
        if terms:
            require(math.isclose(float(loss.detach()), np.mean([r['raw_loss'] for r in terms]),
                                 rel_tol=2e-6, abs_tol=1e-6), 'Actual term losses do not reconcile with native loss')
        capture.update(loss=loss, terms=terms)
        return loss, stats
    adapter.loss_fn.forward = wrapped
    order = np.random.RandomState(42).permutation(len(records)).tolist()
    jsonwrite(out / 'fixed_batch_protocol.json', {'train_ids_in_forward_order': [records[i]['id'] for i in order],
        'gradient_batch_indices': list(range(10)), 'gradient_sample_ids': [[records[i]['id'] for i in order[k*16:(k+1)*16]]
                                                                      for k in range(10)],
        'batch_size': 16, 'sampling': 'All train rows once; RandomState(42) permutation, NOT historical shuffle order.',
        'rng': '420000 + batch index, reset before each forward; same forward graph for both losses.',
        'mode': 'model.train activates frozen original graph/dropout/MLM; model state never updated.'})
    require(cfg['batch_size'] == 16 and cfg['peer_top_k'] == 1 and cfg['peer_lambda'] == .5, 'Frozen config drift')
    print('TOKENIZING_EXISTING_TRAIN', len(records), flush=True)
    data = tensors(records, tokenizer, hierarchy, cfg['text_max_length'])
    def batch(index):
        selected = order[index*16:(index+1)*16]
        return {k: v[selected].cuda() for k, v in data.items() if k != 'fine'}, selected
    terms, batch_rows = [], []
    number_batches = math.ceil(len(records)/16)
    print('FULL_TRAIN_FIXED_CHECKPOINT_FORWARD optimizer_steps=0', flush=True)
    with torch.no_grad():
        for b in range(number_batches):
            seed_batch(b); capture.clear(); adapter.records.clear()
            inputs, indices = batch(b)
            output, mlm, classification = model(**inputs)
            batch_terms = capture['terms']; n = len(batch_terms)
            for term in batch_terms:
                t, neg = term['true_id'], term['negative_id']
                terms.append({'batch_index': b, 'sample_id': records[indices[term['sample_index']]]['id'],
                              'true_label': labels[t]['fine_label'], 'negative_label': labels[neg]['fine_label'],
                              'parent': labels[t]['coarse_label'], 'eligible_terms_in_batch': n,
                              'raw_loss': term['raw_loss'], 'cosine_margin': term['cosine_margin'],
                              'weighted_batch_contribution': cfg['peer_lambda']*term['raw_loss']/n})
            batch_rows.append({'batch_index': b, 'examples': len(indices), 'eligible_terms': n,
                               'actual_peer_loss': float(capture['loss']),
                               'weighted_peer_loss': cfg['peer_lambda']*float(capture['loss'])})
            if (b+1) % 100 == 0 or b+1 == number_batches:
                print('PEER_CONTRIBUTION_FORWARD', b+1, '/', number_batches, flush=True)
            del output, mlm, classification, inputs
    contributions = aggregate_terms(terms, labels, frequencies, number_batches, cfg['peer_lambda'])
    true_rows, parent_rows, negative_rows, pair_rows, contribution_summary = contributions
    expected_terms = int(sum(eligible_mask_for_records(records, hierarchy['parents'])))
    require(len(terms) == expected_terms, 'Full-train term count differs from gold eligibility')
    contribution_summary['train_examples'] = len(records)
    require(math.isclose(contribution_summary['weighted_mean_peer_loss'],
                         np.mean([r['weighted_peer_loss'] for r in batch_rows]), rel_tol=2e-6), 'Batch loss reconciliation failed')
    for name, rows in [('peer_loss_terms_snapshot.csv', terms), ('peer_loss_batches_snapshot.csv', batch_rows),
                       ('peer_loss_by_true_label.csv', true_rows), ('peer_loss_by_parent.csv', parent_rows),
                       ('hard_negative_loss_contribution.csv', negative_rows), ('peer_loss_by_pair_snapshot.csv', pair_rows)]:
        csvwrite(out / name, rows)
    jsonwrite(out / 'peer_loss_contribution_summary.json', contribution_summary)
    gof = HierarchicalGofLoss(hierarchy['depths'], hierarchy['parents'], parent_child_angle_deg=cfg['gof_angle'],
                             length_decay_factor=cfg['gof_decay'], return_components=True).cuda()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    gradient_rows, exclusions, parameter_sets = [], [], []
    print('GRADIENT_CONFLICT_FIXED_10_BATCHES optimizer_steps=0', flush=True)
    for b in range(10):
        seed_batch(b); capture.clear(); adapter.records.clear()
        inputs, _ = batch(b)
        output, mlm, classification = model(**inputs)
        geo, components = gof(model.get_input_embeddings().weight()[-35:-3])
        # Best step is long past the ramp; preserve original trainer's saturated alpha=1.
        alpha = min(1., 8142/max(1, int(10856*cfg['hga_ramp_ratio'])))
        original = mlm + classification + cfg['gof_loss_weight']*alpha*geo
        peer = capture['loss']
        require(bool(torch.isfinite(original) and torch.isfinite(peer)), 'Nonfinite isolated losses')
        require(math.isclose(float(peer.detach()), batch_rows[b]['actual_peer_loss'], rel_tol=2e-5, abs_tol=2e-6),
                'Gradient/snapshot forwards differ under identical batch seed')
        a = torch.autograd.grad(original, [p for _, p in named], allow_unused=True, retain_graph=True)
        bb = torch.autograd.grad(peer, [p for _, p in named], allow_unused=True)
        rows, excluded, included = compare_gradients(named, a, bb, b, cfg['peer_lambda'])
        for row in rows:
            row.update(original_lgsa_loss=float(original.detach()), peer_loss=float(peer.detach()), hga_alpha=alpha)
        gradient_rows.extend(rows); parameter_sets.append(included)
        exclusions.append({'batch_index': b, 'excluded_unshared': excluded})
        require(all(p.grad is None for _, p in named), 'Autograd diagnostic populated parameter .grad')
        print('GRADIENT_CONFLICT_BATCH', b+1, '/ 10', 'full_cosine', rows[0]['cosine'], flush=True)
        del a, bb, original, peer, output, mlm, classification, geo, components, inputs
    require(all(names == parameter_sets[0] for names in parameter_sets), 'Shared gradient parameter set changed across batches')
    scopes = [r for r in gradient_rows if not r['scope'].startswith('BERT_layer_')]
    summary = gradient_summary(scopes)
    require(all(summary[s]['valid_cosines'] == 10 for s in
                ('full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform')), 'Unavailable required gradient scope')
    csvwrite(out / 'gradient_conflict_batches.csv', scopes)
    csvwrite(out / 'gradient_conflict_by_layer.csv', [r for r in gradient_rows if r['scope'].startswith('BERT_layer_')])
    jsonwrite(out / 'gradient_conflict_summary.json', {'scopes': summary, 'checkpoint_epoch': 3,
        'original_objective': 'MLM + classification + gof_loss_weight * alpha(=1) * original HGA/GOF',
        'peer_objective': 'Unweighted native DPT-derived peer; weighted gradient norm also reported for lambda0.5.',
        'shared_definition': 'Trainable parameters connected to BOTH losses; frozen and single-objective-only parameters excluded.',
        'shared_parameter_names': parameter_sets[0], 'excluded_by_batch': exclusions,
        'frozen_raw_prompt_and_keys': True,
        'scope_note': 'Shared MLM anchor transform is a trainable module; frozen raw prompt rows have no parameter gradient.',
        'interpretation': '10 deterministic train batches at best weights, before clipping; not historical trajectory or causal proof.'})
    capture.clear(); adapter.detach()
    require(state_digest(model) == before, 'Checkpoint model state changed during probes')
    require(torch.equal(model.get_input_embeddings().new_embedding.weight[1:29].detach().cpu(), initial), 'Frozen fine rows changed')
    jsonwrite(out / 'model_state_integrity.json', {'before_sha256': before, 'after_sha256': before,
              'all_parameters_and_buffers_exactly_unchanged': True, 'optimizer_instances_created': 0,
              'optimizer_steps': 0, 'frozen_best_and_final_keys_equal_initial': True})
    del model, data, gof
    torch.cuda.empty_cache()
    return contribution_summary, true_rows, parent_rows, negative_rows, pair_rows, summary


def eligible_mask_for_records(records, parents):
    families = defaultdict(set)
    for i in range(28):
        families[parents[i]].add(i)
    return [sum(bool(families[parents[t]]-set(row['labels'])) for t in row['labels']) for row in records]
