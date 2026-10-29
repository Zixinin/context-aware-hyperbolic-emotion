"""Synthetic CPU contracts and real small-file rsync tests, not an Ubuntu run."""
import contextlib,importlib.util,io,json,os,shutil,tempfile,types,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import get_linear_schedule_with_warmup
import mcommon as mc
import mrun,mreport,migrate,verify_migration,github_prepare
from mmodel import GAMMA,install_weight
from a1model import loss_components

spec=importlib.util.spec_from_file_location('frozen_e2_toy_contract',mc.HERE.parent/'exp6e2_calibrated_geodesic_train/tests_local.py')
toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)

class CoordinateToy(toy.ToyModel):
    def __init__(self):
        super().__init__();del self._geodesic_alignment_installed
    def forward(self,input_ids,attention_mask=None,labels=None):
        out=super().forward(input_ids,attention_mask,labels=None)
        c=None if labels is None else loss_components(out['logits'],labels,out['z'],out['decoder_logits'],self.emotion_embeddings)
        return dict(out,components=c,loss=None if c is None else c['total'])

class ScienceTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def test_only_coefficient_changes_exact_original_forward_and_rng(self):
        m=CoordinateToy().eval();b=toy.batch(11);b.pop('ids');state={k:v.clone() for k,v in m.state_dict().items()}
        rng=torch.get_rng_state().clone();a=m(**b);install_weight(m);z=m(**b)
        for k in ['logits','z','decoder_logits']:self.assertTrue(torch.equal(a[k],z[k]))
        for k in ['classification','auxiliary','positive_ce','positive_dot','positive_count']:self.assertTrue(torch.equal(a['components'][k],z['components'][k]))
        self.assertTrue(torch.equal(z['loss'],z['components']['classification']+GAMMA*z['components']['auxiliary']))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(set(state),set(m.state_dict()))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()))
        self.assertNotIn('positive_distance',z['components'])
    def test_aux_vector_scaling_and_bce_paths(self):
        m=install_weight(CoordinateToy()).eval();b=toy.batch(16);b.pop('ids');out=m(**b);c=out['components'];p=list(m.parameters())
        old=torch.autograd.grad(.1*c['auxiliary'],p,allow_unused=True,retain_graph=True)
        new=torch.autograd.grad(GAMMA*c['auxiliary'],p,allow_unused=True,retain_graph=True)
        gb=torch.autograd.grad(c['classification'],p,allow_unused=True)
        for (name,param),a,z,v in zip(m.named_parameters(),old,new,gb):
            if name.startswith('control.classifier_'):self.assertIsNone(a);self.assertIsNone(z);self.assertIsNotNone(v)
            else:self.assertTrue(torch.allclose(z,a*(GAMMA/.1),rtol=5e-5,atol=1e-7))
            if name.startswith('hyper.'):self.assertIsNone(v)
        self.assertIsNone(m.emotion_embeddings.grad)
    def test_multilabel_q_and_coordinate_sign(self):
        m=install_weight(CoordinateToy()).eval();b=toy.batch(11);b.pop('ids');out=m(**b);c=out['components'];y=b['labels']
        q=y/y.sum(-1,keepdim=True)
        expected=(q*(out['z']@m.emotion_embeddings.T)).sum(-1)
        self.assertTrue(torch.equal(c['positive_dot'],expected))
        self.assertTrue(torch.equal(c['auxiliary'],(c['positive_ce']-expected).mean()))
        v=mc.diagnostic_values(out);self.assertAlmostEqual(v['weighted_auxiliary'],v['weighted_ce']+v['weighted_negative_dot'],places=6)
    def test_reject_geodesic_scaled_or_duplicate_models(self):
        with self.assertRaises(RuntimeError):install_weight(toy.ToyModel())
        m=CoordinateToy();m.hyper._exp6c1s_scale_installed=True
        with self.assertRaises(RuntimeError):install_weight(m)
        m=install_weight(CoordinateToy())
        with self.assertRaises(RuntimeError):install_weight(m)
    def test_actual_two_steps_match_original_manual_training_contract(self):
        torch.manual_seed(1);a=install_weight(CoordinateToy()).eval()
        torch.manual_seed(1);b=install_weight(CoordinateToy()).eval()
        oa=torch.optim.AdamW(a.parameters(),lr=5e-5);ob=torch.optim.AdamW(b.parameters(),lr=5e-5)
        sa=get_linear_schedule_with_warmup(oa,765,7648);sb=get_linear_schedule_with_warmup(ob,765,7648)
        for n in [16,11]:
            batch=toy.batch(n);manual={k:v for k,v in batch.items() if k!='ids'}
            loss=b(**manual)['loss'];(loss*n).backward()
            for p in b.parameters():
                if p.grad is not None:p.grad.div_(n)
            expected_norm=torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
            ob.step();sb.step();ob.zero_grad(set_to_none=True)
            actual=mrun.step(a,dict(batch),oa,sa,detailed=True)
            self.assertAlmostEqual(actual['unclipped_grad_norm'],float(expected_norm),places=6)
            self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
    def test_frozen_code_and_historical_artifact_names(self):
        reference=mc.read(mc.HERE/'INPUT_REFERENCE.json')['frozen_source_directories'];self.assertEqual(len(reference),8)
        for name,expected in reference.items():
            self.assertEqual({p.name:mc.sha(p) for p in (mc.HERE.parent/name).iterdir() if p.is_file()},expected)
        source=(mc.HERE/'mrun.py').read_text()
        self.assertIn("A1/'INITIALIZATION_PARITY.json'",source)
        self.assertNotIn("A1/'EXP6E3_INITIALIZATION_PARITY.json'",source)
        self.assertIn('backup_preflight(root)',source)
        with self.assertRaises(RuntimeError):mrun.predict(None,None,[None]*3821,'No Test')
    def test_four_model_report_keeps_negative_results(self):
        names=(mc.REPO/'数据集/GoEmotions/metadata/emotions.txt').read_text().splitlines();names=[n for n in names if n!='neutral']
        gold=mc.np.eye(27,dtype=mc.np.float32).repeat(2,axis=0);ids=mc.np.array(['fixture'+str(i) for i in range(len(gold))])
        manifest=dict(names27=names,splits={'validation':dict(retained_ids=ids.tolist())})
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/n for n in ['a0','a1','control','geodesic']]
            for path,logits in zip(paths,[gold*6-3,gold*6-3,mc.np.zeros_like(gold),gold*6-3]):
                path.mkdir();mc.np.savez(path/'validation_predictions.npz',logits=logits,targets=gold,ids=ids,label_names=mc.np.array(names))
                mc.write(path/'validation_metrics.json',mc.extended_score(gold,mc.probabilities(logits),names)[0])
            root=paths[2]
            mc.table(root/'training_log.csv',[dict(epoch=e,training_first_text_clip_fraction=1.,training_later_text_clip_fraction=.5,training_final_fraction_above099=.9,
                training_final_radius_mean=.993,training_final_radius_min=.9,training_final_radius_max=.996,global_gradient_clip_fraction=.4,gradient_norm_mean=1.,gradient_norm_max=3.) for e in range(1,5)])
            for tag in ['fresh_initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
                p=root/'probes'/tag;p.mkdir(parents=True)
                mc.table(p/'MODULE_GRADIENTS.csv',[dict(module='shared_encoder',matched_aux_to_bce=.1,bce_matched_aux_cosine=-.1,ce_negative_dot_cosine=.2)])
            with patch.multiple(mreport,A0=paths[0],A1=paths[1],G2=paths[3]):mreport.report(root,manifest)
            self.assertEqual(len(mc.csvrows(root/'ALL_MODELS_PER_LABEL.csv')),108)
            self.assertEqual(len(mc.csvrows(root/'ALL_PAIRED_LABEL_DELTAS.csv')),108)
            text=(root/'EXP6E3_TRAINING_SUMMARY_CN.md').read_text();self.assertIn('A_lower_gamma',text);self.assertIn('B_geodesic_vs_coordinate_same_gamma',text)
            self.assertIn('同gamma并非同梯度强度',text)

class BackupTests(unittest.TestCase):
    def test_source_log_is_closed_before_copying_static_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'contract.sh'
            script.write_text('''set -Eeuo pipefail
exec 3>&1
mkfifo "$1/science.pipe"
tee "$1/source.log" < "$1/science.pipe" >&3 &
logpid=$!
exec > "$1/science.pipe" 2>&1
echo SCIENCE_COMPLETE
exec > "$1/post.log" 2>&1
wait "$logpid"
test "$(cat "$1/source.log")" = SCIENCE_COMPLETE
echo LATER_BACKUP_OUTPUT
test "$(cat "$1/source.log")" = SCIENCE_COMPLETE
echo QUIESCENT_SOURCE_PASS >&3
''')
            result=subprocess.run(['bash',str(script),str(root)],text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'source.log').read_text(),'SCIENCE_COMPLETE\n')
            self.assertEqual((root/'post.log').read_text(),'LATER_BACKUP_OUTPUT\n')
            self.assertIn('QUIESCENT_SOURCE_PASS',result.stdout)
    def test_existing_directory_on_ubuntu_root_is_not_a_mounted_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            media=Path(tmp)/'looks_like_external';media.mkdir()
            response=dict(filesystems=[dict(target=str(media),source='/dev/fixture',fstype='ext4',options='rw',**{'maj:min':'0:0'})])
            with patch.multiple(migrate,MEDIA=media,DEST=media/'destination'),patch.object(migrate.platform,'system',return_value='Linux'),\
                patch.object(migrate.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=json.dumps(response),stderr='')):
                with self.assertRaisesRegex(RuntimeError,'root filesystem'):migrate.mount()
            self.assertFalse((media/'destination').exists())
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);src=base/'repo';src.mkdir();(src/'.git').mkdir();(src/'.git/HEAD').write_text('ref: refs/heads/main')
            for name in ['weights.pt','predictions.npz','labels.npy','metrics.csv','run.log','config.json']:(src/name).write_bytes(('fixture '+name).encode())
            (src/'.venv').mkdir();(src/'.venv/not_needed.py').write_text('environment')
            extra=base/'dependency';extra.mkdir();(extra/'embedding.bin').write_bytes(b'fixed labels')
            (src/'linked_labels.bin').symlink_to(extra/'embedding.bin')
            media=base/'media';media.mkdir();dest=media/'owned_backup';records=base/'records';records.mkdir()
            stamp=dict(target=str(media),source='/dev/fixture',fstype='fixture',options='rw',major_minor='9:9',st_dev=999,media=str(media))
            with patch.multiple(migrate,REPO=src,MEDIA=media,DEST=dest,RECORDS=records),patch.object(migrate,'mount',return_value=stamp),\
                patch.object(migrate,'roots',return_value={'repository':src,'external/dependency':extra}),patch.object(migrate,'required_check'),\
                patch.object(migrate.shutil,'disk_usage',return_value=types.SimpleNamespace(free=100*1024**3)):
                yield src,extra,dest,records
    @unittest.skipUnless(shutil.which('rsync'),'Native rsync unavailable')
    def test_full_copy_arrays_checkpoints_git_links_and_target_only_validation(self):
        with self.fixture() as (src,extra,dest,records),contextlib.redirect_stdout(io.StringIO()):
            migrate.sync(records/'one','pre',src)
            self.assertTrue((dest/'payload/repository/.git/HEAD').exists())
            for n in ['weights.pt','predictions.npz','labels.npy','metrics.csv','run.log']:self.assertEqual((dest/'payload/repository'/n).read_bytes(),(src/n).read_bytes())
            self.assertFalse((dest/'payload/repository/.venv').exists())
            link=dest/'payload/repository/linked_labels.bin';self.assertFalse(link.is_symlink());self.assertEqual(link.read_bytes(),b'fixed labels')
            verify_migration.verify(dest,records/'one/FILES.json')
            link.write_bytes(b'bad destination')
            with self.assertRaises(RuntimeError):verify_migration.verify(dest,records/'one/FILES.json')
    @unittest.skipUnless(shutil.which('rsync'),'Native rsync unavailable')
    def test_incremental_retains_previous_versions_and_never_deletes_extra_files(self):
        with self.fixture() as (src,extra,dest,records),contextlib.redirect_stdout(io.StringIO()):
            migrate.sync(records/'one','pre',src)
            original=(src/'run.log').read_bytes();(src/'run.log').write_bytes(b'new scientific log')
            (dest/'payload/repository/older_project_file').write_text('preserve')
            migrate.sync(records/'two','pre',src)
            self.assertEqual((dest/'versions/two/repository/run.log').read_bytes(),original)
            self.assertEqual((dest/'payload/repository/older_project_file').read_text(),'preserve')
    def test_mount_failure_and_low_space_stop_before_destination_creation(self):
        with self.fixture() as (src,extra,dest,records):
            with patch.object(migrate,'mount',side_effect=RuntimeError('not mounted')):
                with self.assertRaises(RuntimeError):migrate.sync(records/'one','pre',src)
            self.assertFalse(dest.exists())
            with patch.object(migrate.shutil,'disk_usage',return_value=types.SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):migrate.sync(records/'two','pre',src)
            self.assertFalse(dest.exists())
    def test_unknown_existing_destination_and_unreviewed_symlink_rejected(self):
        with self.fixture() as (src,extra,dest,records):
            dest.mkdir()
            with self.assertRaises(RuntimeError):migrate.sync(records/'one','pre',src)
            outside=records/'not_reviewed';outside.write_text('outside scope');(src/'bad_link').symlink_to(outside)
            with self.assertRaises(RuntimeError):migrate.inventory({'repository':src,'external/dependency':extra})
    def test_rsync_failure_cannot_emit_success_receipt(self):
        with self.fixture() as (src,extra,dest,records):
            with patch.object(migrate.subprocess,'run',return_value=types.SimpleNamespace(returncode=23)):
                with self.assertRaises(RuntimeError):migrate.sync(records/'one','pre',src)
            self.assertFalse((records/'one/VERIFICATION.json').exists())
    def test_source_change_or_symlink_destination_not_silently_accepted(self):
        with self.fixture() as (src,extra,dest,records):
            files,_,_=migrate.inventory({'repository':src,'external/dependency':extra})
            (src/'run.log').write_text('changed')
            with self.assertRaises(RuntimeError):migrate.source_still_stable(files)
            dest.mkdir();(dest/'payload').symlink_to(src)
            with self.assertRaises(RuntimeError):migrate.safe_target('payload/repository/run.log')
    def test_github_staging_does_not_modify_source_and_stops_on_credentials(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(tmp);repo=base/'repo';source=repo/'experiments/exp6_fixture';source.mkdir(parents=True)
            (repo/'results').mkdir();(source/'main.py').write_text('print("safe")')
            (source/'weights.pt').write_bytes(b'binary')
            rows=github_prepare.prepare(repo,base/'stage')
            self.assertEqual(len(rows),1);self.assertFalse((repo/'.git').exists())
            self.assertFalse((base/'stage/files/experiments/exp6_fixture/weights.pt').exists())
            (source/'credential.txt').write_text('ghp_'+'A'*36)
            with self.assertRaises(RuntimeError):github_prepare.prepare(repo,base/'unsafe')
            self.assertFalse((base/'unsafe').exists())

if __name__=='__main__':unittest.main(verbosity=2)
