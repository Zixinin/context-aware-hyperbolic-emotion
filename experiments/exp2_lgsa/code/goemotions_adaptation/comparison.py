"""Source-labelled Experiment 1/2 comparison, generated only from real results."""
import csv
from common import PACKAGE, read, write, sha, require


def compare(out):
    base = PACKAGE / 'baseline_reference'
    reference = read(base / 'reference_manifest.json')
    for name in ['test_metrics.json','rare_label_metrics.json','test_per_label_fixed_0.30.csv']:
        require(sha(base / name) == reference['artifacts_sha256'][name], 'Baseline reference changed')
    baseline = read(base / 'test_metrics.json')
    rare = read(base / 'rare_label_metrics.json')['test']
    summaries = []
    for setting, prefix in [('fixed_0.30','test_fixed_0.30'), ('validation_selected','test_validation_selected')]:
        actual = read(out / (prefix + '_metrics.json'))
        former = dict(baseline[setting], rare_mean_f1=rare[setting]['mean_rare_label_f1'],
                      nonrare_mean_f1=rare[setting]['mean_non_rare_label_f1'])
        for metric in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','rare_mean_f1','nonrare_mean_f1']:
            summaries.append({'setting':setting,'metric':metric,'baseline_percent':100*former[metric],
                'lgsa_percent':100*actual[metric], 'delta_percentage_points':100*(actual[metric]-former[metric]),
                'baseline_threshold':former['threshold'],'lgsa_threshold':actual['threshold']})
    def csv_write(path, records):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    csv_write(out/'comparison_summary.csv',summaries)
    old={int(r['label_id']):r for r in csv.DictReader((base/'test_per_label_fixed_0.30.csv').open())}
    new=list(csv.DictReader((out/'test_fixed_0.30_per_label.csv').open()))
    deltas=[]
    for row in new:
        i=int(row['label_id']); require(int(row['support'])==int(old[i]['support']),'Test label support mismatch')
        deltas.append(dict(row,baseline_f1_percent=100*float(old[i]['f1']),lgsa_f1_percent=100*float(row['f1']),
                           delta_percentage_points=100*(float(row['f1'])-float(old[i]['f1']))))
    csv_write(out/'per_label_delta.csv',deltas)
    difficult={'grief','relief','realization','nervousness','disappointment','remorse'}
    csv_write(out/'difficult_labels.csv',[r for r in deltas if r['label'] in difficult])
    write(out/'comparison_provenance.json',{'baseline_job_id':'2968116','baseline_source':reference,
        'settings':'Each model uses its own validation-selected threshold for Setting B; test never selects it.',
        'limits':'Single seed, 4 epochs. Loss, prompt, GAT and optimizer differ; improvement cannot establish geometry-only causation.'})
