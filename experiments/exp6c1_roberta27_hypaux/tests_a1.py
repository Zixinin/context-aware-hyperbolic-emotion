"""CPU-only scientific contract tests. No real RoBERTa or GPU training here."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from a1lib import HERE, REPO, EXP6B, CONFIG, check_sources, read, write, sha, seal, check_seal
from a1model import A1Model, loss_components, build
from model27 import Roberta27, seed_everything
from data27 import loader, targets, load_dataset
from metrics27 import save_predictions
from metrics27 import write_csv

NAMES = ["grief", "pride", "relief", "nervousness", "embarrassment", "remorse", "fear"] + [f"emotion{i}" for i in range(20)]


def reference_loss():
    paths = [EXP6B / "tools/approved_multilabel_loss.py",
             REPO.parent / "交付/Exp6B_HypEmo_20261019/stage_b_seed42_v1/approved_multilabel_loss.py"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise RuntimeError("Missing read-only verified Exp6B reference for CPU parity tests")
    assert sha(path) == read(HERE / "EXP6B_REFERENCE.json")["approved_loss_source_sha256"]
    spec = importlib.util.spec_from_file_location("a1_test_reference_formula", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.loss_components


class LossTests(unittest.TestCase):
    def inputs(self, batch=3):
        torch.manual_seed(7)
        logits = torch.randn(batch, 27, requires_grad=True)
        targets = torch.zeros(batch, 27)
        for i in range(batch):
            targets[i, :i % 3 + 1] = 1
        z = (torch.randn(batch, 100) * .02).requires_grad_()
        raw = torch.randn(batch, 27, requires_grad=True)
        vectors = torch.randn(27, 100) * .02
        return logits, targets, z, raw, vectors

    def test_exact_validated_formula_a_value_and_gradients_without_weighted_bce(self):
        x = self.inputs()
        new = loss_components(*x)
        old = reference_loss()(torch.cat([x[0], torch.zeros(3, 1)], 1),
                               torch.cat([x[1], torch.zeros(3, 1)], 1), *x[2:])
        torch.testing.assert_close(new["auxiliary"], old["auxiliary"], rtol=0, atol=0)
        torch.testing.assert_close(new["classification"], nn.functional.binary_cross_entropy_with_logits(x[0], x[1]), rtol=0, atol=0)
        for variable in [x[2], x[3]]:
            a = torch.autograd.grad(new["auxiliary"], variable, retain_graph=True)[0]
            b = torch.autograd.grad(old["auxiliary"], variable, retain_graph=True)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(new["total"], new["classification"] + .1 * new["auxiliary"], rtol=0, atol=0)

    def test_positive_mean_and_batch_mean_are_not_sums_or_argmax(self):
        x = self.inputs(1)
        x[1][0, [1, 2]] = 1
        out = loss_components(*x)
        expected = -nn.functional.log_softmax(x[3], -1)[0, :3].mean() - (x[2] @ x[4].T)[0, :3].mean()
        torch.testing.assert_close(out["auxiliary"], expected)
        repeated = [t.repeat(5, 1) for t in x[:4]] + [x[4]]
        rep = loss_components(*repeated)
        torch.testing.assert_close(out["auxiliary"], rep["auxiliary"])
        torch.testing.assert_close(out["classification"], rep["classification"])

    def test_auxiliary_gradient_paths_and_fixed_labels(self):
        x = self.inputs()
        out = loss_components(*x)
        self.assertIsNone(torch.autograd.grad(out["auxiliary"], x[0], allow_unused=True, retain_graph=True)[0])
        self.assertIsNone(torch.autograd.grad(out["classification"], x[2], allow_unused=True, retain_graph=True)[0])
        self.assertGreater(float(torch.autograd.grad(out["auxiliary"], x[2], retain_graph=True)[0].norm()), 0)
        self.assertGreater(float(torch.autograd.grad(out["auxiliary"], x[3], retain_graph=True)[0].norm()), 0)
        vectors = x[4].clone().requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "fixed"):
            loss_components(*x[:4], vectors)

    def test_negative_auxiliary_is_preserved_without_clamp(self):
        logits, gold = torch.zeros(1, 27), torch.zeros(1, 27)
        gold[0, 0] = 1
        z, e = torch.zeros(1, 100), torch.zeros(27, 100)
        z[0, 0] = e[0, 0] = .8
        raw = torch.full((1, 27), -20.)
        raw[0, 0] = 20
        out = loss_components(logits, gold, z, raw, e)
        self.assertLess(float(out["auxiliary"]), 0)
        self.assertAlmostEqual(float(out["auxiliary"]), -.64, places=6)

    def test_empty_target_outside_ball_and_nonfinite_stop(self):
        for case in ["empty", "outside", "nan", "28labels"]:
            x = list(self.inputs())
            if case == "empty":
                x[1][0] = 0
            elif case == "outside":
                x[2] = torch.ones(3, 100)
            elif case == "nan":
                x[3] = torch.full((3, 27), float("nan"))
            else:
                x[0] = torch.zeros(3, 28)
                x[1] = torch.ones(3, 28)
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                loss_components(*x)


class ToyConfig:
    hidden_size = 8
    def to_dict(self):
        return {"hidden_size": 8}
    def save_pretrained(self, path):
        write(Path(path) / "config.json", self.to_dict())


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.config = ToyConfig()
        self.calls = 0
    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 27)
    def decode(self, z):
        return self.linear(z)


class ToyHyper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 100)
        self.decoder = ToyDecoder()
    def encode(self, h):
        return torch.tanh(self.encoder(h)) * .03


class ToyTokenizer:
    vocab_size = 10
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    init_kwargs = {'synthetic_CPU_only':True}
    def __call__(self, texts, **kwargs):
        return dict(input_ids=torch.tensor([[0, 3, 2] for _ in texts]),
                    attention_mask=torch.ones(len(texts), 3, dtype=torch.long))


def toy_load(_snapshot):
    return Roberta27(ToyEncoder()), ToyTokenizer(), {"missing_keys": []}


def toy_hyper(_names):
    return ToyHyper(), torch.randn(27, 100) * .02, {"synthetic_CPU_only": True}


class ModelTests(unittest.TestCase):
    def test_single_encoder_pass_exact_A0_forward_and_auxiliary_encoder_gradients(self):
        seed_everything()
        control = Roberta27(ToyEncoder())
        hyper = ToyHyper()
        model = A1Model(control, hyper, torch.randn(27, 100) * .02)
        ids, mask = torch.tensor([[0, 3, 2]]), torch.ones(1, 3, dtype=torch.long)
        labels = torch.zeros(1, 27)
        labels[0, [0, 2]] = 1
        model.train()
        torch.manual_seed(123)
        expected = control(ids, mask)["logits"]
        calls = control.encoder.calls
        torch.manual_seed(123)
        actual = model(ids, mask, labels)
        self.assertEqual(control.encoder.calls, calls + 1)
        torch.testing.assert_close(expected, actual["logits"], rtol=0, atol=0)
        self.assertEqual(len(control.encoder._forward_hooks), 0)
        actual["components"]["auxiliary"].backward()
        self.assertGreater(float(control.encoder.embedding.weight.grad.norm()), 0)
        self.assertIsNone(control.classifier_weight.grad)
        self.assertIsNone(model.emotion_embeddings.grad)

    def test_hook_is_removed_if_forward_raises(self):
        control = Roberta27(ToyEncoder())
        model = A1Model(control, ToyHyper(), torch.randn(27, 100) * .02)
        with patch.object(control.encoder, "forward", side_effect=RuntimeError("toy failure")):
            with self.assertRaisesRegex(RuntimeError, "toy failure"):
                model(torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
        self.assertEqual(len(control.encoder._forward_hooks), 0)


class IntegrityTests(unittest.TestCase):
    def test_frozen_A0_sources_and_A1_package(self):
        source = check_sources()
        self.assertEqual(len(source["a0"]), 15)

    def test_standard_library_inspector_import_under_no_site_packages(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import inspect_a1;assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules;print('STDLIB_OK')"
        r = subprocess.run([sys.executable, "-S", "-c", code, str(HERE)], text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("STDLIB_OK", r.stdout)

    def test_seal_rejects_tampering_and_never_replaces_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "metrics.json", {"v": 1})
            seal(root, "SUCCESS.json", status="PASS")
            check_seal(root, "SUCCESS.json")
            with self.assertRaises(RuntimeError):
                seal(root, "SUCCESS.json", status="again")
            write(root / "metrics.json", {"v": 2})
            with self.assertRaises(RuntimeError):
                check_seal(root, "SUCCESS.json")

    def test_real_canonical_27_dataset_and_budget(self):
        data, manifest = load_dataset()
        self.assertEqual([len(data[s]) for s in ["train", "validation", "test"]], [30587, 3834, 3821])
        self.assertEqual([manifest["splits"][s]["multi_emotion"] for s in data], [5767, 712, 667])
        self.assertFalse(CONFIG["classification_distance_weighting"])
        self.assertEqual(CONFIG["auxiliary_weight"], .1)


class SmokeFlowTests(unittest.TestCase):
    def test_full_smoke_control_flow_with_tiny_CPU_toy_and_historical_anchor(self):
        import a1run
        data = {}
        for split, count in [("train", 48), ("validation", 64), ("test", 32)]:
            data[split] = []
            for i in range(count):
                labels = [0] if i < 16 else [1, 2] if i < 32 else [3]
                original = labels if i < 32 else [27, 3]
                data[split].append(dict(id=split + str(i), text=str(i), labels=labels, original_labels=original))
        manifest = dict(protocol="synthetic CPU fixture only", names27=NAMES, neutral_original_index=27,
            mapping=[dict(new_index=i, original_index=i, emotion=n) for i, n in enumerate(NAMES)],
            text_overlap_note="synthetic", text_overlap_audit={}, splits={s: dict(original_count=len(rows),
                retained_count=len(rows), removed_neutral_only=0,
                single_emotion=sum(len(r["labels"]) == 1 for r in rows), multi_emotion=sum(len(r["labels"]) > 1 for r in rows),
                multi_fraction=sum(len(r["labels"]) > 1 for r in rows) / len(rows), mean_positive_labels=float(targets(rows).sum(1).mean()),
                retained_neutral_plus_emotion=sum(27 in r["original_labels"] for r in rows), positive_support=targets(rows).sum(0).astype(int).tolist())
                for s, rows in data.items()})
        real_to, real_zeros = torch.Tensor.to, torch.zeros
        def cpu_to(tensor, *args, **kwargs):
            if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
                args = ("cpu", *args[1:])
            return real_to(tensor, *args, **kwargs)
        def cpu_zeros(*args, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return real_zeros(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root, baseline = Path(tmp) / "run_toy", Path(tmp) / "A0"
            root.mkdir()
            (baseline / "smoke").mkdir(parents=True)
            snapshot = Path(tmp) / ("a" * 40)
            snapshot.mkdir()
            write(snapshot / "config.json", {"synthetic_only": True})
            write(baseline / "plan.json", dict(snapshot=str(snapshot), snapshot_files_sha256={"config.json": sha(snapshot / "config.json")}))
            with patch("torch.cuda.get_rng_state", side_effect=lambda *a: torch.zeros(16, dtype=torch.uint8)), \
                    patch("torch.cuda.set_rng_state"), patch("torch.Tensor.to", new=cpu_to), patch("torch.zeros", new=cpu_zeros):
                seed_everything()
                old_model, tokenizer, _ = toy_load(snapshot)
                write(baseline/'model_identity.json',a1run.a0_model_identity(snapshot,old_model,tokenizer,{'missing_keys':[]}))
                torch.save(old_model.state_dict(), baseline / "smoke/smoke_weights.pt")
                ids = next(iter(loader(data["train"], tokenizer, shuffle=True)))["ids"]
                write(baseline / "epoch1_first_batch.json", dict(ids=ids))
                del old_model
                with patch.object(a1run, "A0_RUN", baseline), patch.object(a1run, "run_path", side_effect=Path), \
                        patch.object(a1run, "runtime", return_value={"gpu": "synthetic CPU only"}), \
                        patch.object(a1run, "baseline_evidence", return_value=(data, manifest, {"synthetic": True})), \
                        patch.object(a1run, "verify_upstream", return_value={}), \
                        patch.object(a1run, "protected_inventory", return_value={"synthetic_protected": "sha"}), \
                        patch.object(a1run, "load_a0_model", side_effect=toy_load), \
                        patch("a1model.load_a0_model", side_effect=toy_load), \
                        patch("a1model.build_hyperbolic", side_effect=toy_hyper), \
                        patch.object(a1run, "load_reference_loss", side_effect=reference_loss), \
                        contextlib.redirect_stdout(io.StringIO()) as captured:
                    a1run.smoke(root)
                self.assertIn("EXP6C1_SMOKE_VERIFIED", captured.getvalue())
                self.assertNotIn(a1run.SUCCESS, captured.getvalue())
                self.assertFalse((root / "FORMAL_TEST_CLAIM.json").exists())
                report = read(root / "smoke/SMOKE_REPORT.json")
                self.assertTrue(report["initialization"]["initial_training_logits_exact_equal"])
                self.assertEqual(report["reload"]["max_abs_logit_error"], 0)
                self.assertEqual(report["formal_epochs"], 0)
                self.assertEqual(report["test_inference_calls"], 0)
                check_seal(root / "smoke", "SMOKE_VERIFIED.json")


class SavedResultTests(unittest.TestCase):
    """Exercise four-epoch saved-artifact validation without actual training."""
    def test_complete_result_and_reject_test_reuse_selection_and_loss_tampering(self):
        import a1run
        data = dict(train=[dict(id=f'train{i}', text='x', labels=[0]) for i in range(30587)],
                    validation=[dict(id=f'val{i}', text='x', labels=[0]) for i in range(2)],
                    test=[dict(id=f'test{i}', text='x', labels=[0]) for i in range(2)])
        manifest = dict(names27=NAMES)
        baseline = {'synthetic':True}
        source = {'synthetic_source':'hash'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'plan.json', dict(source_sha256=source, baseline=baseline, snapshot='/synthetic', snapshot_files_sha256={}))
            write(root/'config.json', CONFIG)
            write(root/'dataset_manifest.json', manifest)
            write(root/'protected_before.json', {'synthetic':'hash'})
            env = dict(completed_epochs=4, completed_optimizer_steps=7648, formal_test_inference_calls=1)
            write(root/'environment.json', env)
            (root/'best_checkpoint').mkdir()
            (root/'best_checkpoint/model.pt').write_bytes(b'synthetic weights, not a model')
            digest = sha(root/'best_checkpoint/model.pt')
            logits = np.full((2,27), -4, dtype=np.float32)
            logits[:,0] = 4
            rares = {}
            for split in ['validation','test']:
                metrics, rares[split] = save_predictions(root, split, logits, targets(data[split]),
                                                        [r['id'] for r in data[split]], NAMES)
                write(root/f'{split}_loss_diagnostics.json', dict(loss=.3,bce=.2,auxiliary=1))
            write(root/'rare_label_metrics.json', rares)
            selection = dict(epoch=2, global_step=3824, validation_macro_f1=metrics['macro_f1'], threshold=.30,
                             selection_split='validation', checkpoint_sha256=digest)
            write(root/'checkpoint_selection.json', selection)
            write(root/'optimizer_protocol.json', dict(total_steps=7648,warmup_steps=765,max_grad_norm=1.0))
            history=[dict(epoch=i,global_step=i*1912,training_examples=30587,
                          validation_macro_f1=0 if i==1 else metrics['macro_f1']) for i in range(1,5)]
            write_csv(root/'training_log.csv',history)
            rows=[dict(step=i,loss=.3,bce=.2,auxiliary=1,max_ball_norm=.5,lr_after=0) for i in range(1,7649)]
            write_csv(root/'batch_log.csv',rows)
            for epoch in range(1,5):
                write(root/f'train_order_epoch{epoch}.json',[r['id'] for r in data['train']])
                write(root/f'epoch{epoch}_first_batch.json',dict(gradients={k:dict(norm=1) for k in
                    ['shared_encoder','classifier','hyperbolic_projection','auxiliary_decoder']}))
            write(root/'checkpoint_reload_verification.json',dict(num_examples=3834,ids_equal=True,targets_equal=True,metrics_equal=True,threshold_predictions_equal=True))
            claim=dict(call=1,threshold=.30,checkpoint_sha256=digest)
            write(root/'FORMAL_TEST_CLAIM.json',claim)
            with patch.object(a1run,'run_path',side_effect=Path), patch.object(a1run,'check_sources',return_value=source), \
                    patch.object(a1run,'baseline_evidence',return_value=(data,manifest,baseline)), \
                    patch.object(a1run,'protected_inventory',return_value={'synthetic':'hash'}), \
                    patch.object(a1run.a0,'snapshot_inventory',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                a1run.verify(root,sealed=False)
                write(root/'checkpoint_selection.json',dict(selection,epoch=3,global_step=5736))
                with self.assertRaisesRegex(RuntimeError,'selected checkpoint'):
                    a1run.verify(root,sealed=False)
                write(root/'checkpoint_selection.json',selection)
                write(root/'environment.json',dict(env,formal_test_inference_calls=2))
                with self.assertRaisesRegex(RuntimeError,'formal runtime incomplete'):
                    a1run.verify(root,sealed=False)
                write(root/'environment.json',env)
                rows[-1]['loss']=.8
                write_csv(root/'batch_log.csv',rows)
                with self.assertRaisesRegex(RuntimeError,'BCE\\+0.1aux'):
                    a1run.verify(root,sealed=False)
                rows[-1]['loss']=.3
                write_csv(root/'batch_log.csv',rows)
                write(root/'FORMAL_TEST_CLAIM.json',dict(claim,call=2))
                with self.assertRaisesRegex(RuntimeError,'Single formal test claim'):
                    a1run.verify(root,sealed=False)


class FormalOrchestrationTests(unittest.TestCase):
    def test_four_epoch_loop_best_reload_single_test_summary_with_no_training_stubs(self):
        """Full real loop/CSV/selection/export code; all ML updates/inference stubbed."""
        import a1run
        from a1lib import state_hash
        data={s:[dict(id=s+str(i),text='toy',labels=[0]) for i in range(n)]
              for s,n in [('train',30587),('validation',3834),('test',3821)]}
        manifest={'names27':NAMES}
        class FakeTrainingLoader:
            def __len__(self):return 1912
            def __iter__(self):
                for start in range(0,30587,16):
                    yield {'ids':[r['id'] for r in data['train'][start:start+16]]}
        def fake_loader(rows,tokenizer,shuffle=False):
            return FakeTrainingLoader() if shuffle else rows
        def fake_step(model,batch,optimizer,scheduler,detailed=False):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ids=batch['ids']
            return dict(ids=ids,batch_size=len(ids),loss=.3,bce=.2,auxiliary=1.,weighted_auxiliary=.1,
                weighted_auxiliary_to_bce=.5,absolute_weighted_auxiliary_to_bce=.5,
                positive_ce_mean=1.1,positive_dot_mean=.1,max_ball_norm=.5,unclipped_grad_norm=.2,
                logits_shape=[len(ids),27],targets_shape=[len(ids),27],
                gradients={k:dict(norm=1.) for k in ['shared_encoder','classifier','hyperbolic_projection','auxiliary_decoder']} if detailed else None)
        calls=[]
        def fake_predict(model,rows,description):
            calls.append(description)
            gold=targets(rows)
            logs=np.full(gold.shape,-4,dtype=np.float32)
            # Epoch2 and3 tie; checkpoint must remain at epoch2. Reload/test
            # reproduce that selected inference instead of the last epoch.
            if len(calls) in [2,3,5,6]:logs[:,0]=4
            ids=np.array([r['id'] for r in rows],dtype=str)
            return logs,gold,ids,.3,dict(loss=.3,bce=.2,auxiliary=1.,weighted_auxiliary=.1)
        real_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            if args and isinstance(args[0],(str,torch.device)) and str(args[0]).startswith('cuda'):
                args=('cpu',*args[1:])
            return real_to(tensor,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root,old=Path(tmp)/'run_toy',Path(tmp)/'A0'
            root.mkdir();old.mkdir()
            snapshot=Path(tmp)/('b'*40);snapshot.mkdir()
            write(snapshot/'config.json',{'synthetic':True})
            snapshot_hash={'config.json':sha(snapshot/'config.json')}
            for epoch in range(1,5):write(old/f'epoch{epoch}_first_batch.json',{'ids':[r['id'] for r in data['train'][:16]]})
            oldlogs=np.full((3821,27),-4,dtype=np.float32);oldlogs[:,0]=4
            save_predictions(old,'test',oldlogs,targets(data['test']),[r['id'] for r in data['test']],NAMES)
            with patch('torch.cuda.get_rng_state',side_effect=lambda *a:torch.zeros(16,dtype=torch.uint8)), \
                    patch('torch.cuda.set_rng_state'),patch('torch.Tensor.to',new=cpu_to), \
                    patch('a1model.load_a0_model',side_effect=toy_load),patch('a1model.build_hyperbolic',side_effect=toy_hyper):
                seed_everything()
                model,tokenizer,_,identity=build(snapshot,NAMES)
                write(root/'INITIALIZATION_PARITY.json',{'shared_state_sha256':state_hash(model.control.state_dict()),
                    'hyperbolic_initial_state_sha256':state_hash(model.hyper.state_dict())})
                write(root/'hyperbolic_identity.json',identity)
                np.save(root/'fixed_emotion_embeddings.npy',model.emotion_embeddings.numpy())
                source=check_sources()
                plan={'runtime':{'gpu':'synthetic CPU only'},'source_sha256':source,'snapshot_files_sha256':snapshot_hash}
                write(root/'protected_before.json',{'synthetic':'hash'})
                with patch.object(a1run,'A0_RUN',old),patch.object(a1run,'run_path',side_effect=Path), \
                        patch.object(a1run,'assert_plan',return_value=(data,manifest,plan,snapshot)), \
                        patch.object(a1run,'loader',side_effect=fake_loader),patch.object(a1run,'train_step',side_effect=fake_step), \
                        patch.object(a1run,'predict',side_effect=fake_predict),patch.object(a1run,'protected_inventory',return_value={'synthetic':'hash'}), \
                        patch.object(a1run,'verify') as verifier, \
                        patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
                        patch.object(ToyTokenizer,'save_pretrained',create=True,side_effect=lambda p:write(Path(p)/'tokenizer.json',{'toy':True})), \
                        contextlib.redirect_stdout(io.StringIO()) as captured:
                    a1run.train(root)
                self.assertEqual(verifier.call_count,1)
                self.assertEqual(len(calls),6)
                self.assertEqual(sum('formal test' in s for s in calls),1)
                self.assertEqual(read(root/'checkpoint_selection.json')['epoch'],2)
                self.assertEqual(read(root/'environment.json')['completed_optimizer_steps'],7648)
                self.assertTrue((root/'A0_vs_A1_27_label_metrics.csv').is_file())
                self.assertIn('Stop after Exp6C-1',(root/'EXP6C1_SUMMARY.md').read_text())
                self.assertIn(a1run.SUCCESS,captured.getvalue())
                check_seal(root,'SUCCESS.json')


if __name__ == "__main__":
    unittest.main(verbosity=2)
