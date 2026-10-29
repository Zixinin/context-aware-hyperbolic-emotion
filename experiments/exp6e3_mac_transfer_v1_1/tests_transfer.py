"""Targeted tests for changed policy, archive integrity and safe Mac extraction."""
import hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import transfer_policy as policy
import export_results as export
import mac_unpack_verify as unpack

class TransferTests(unittest.TestCase):
    def test_policy_requires_exact_new_authorization_and_never_backup_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):policy.check_policy(root)
            (root/'USER_AUTHORIZED_TRANSFER_POLICY.json').write_text(json.dumps(policy.POLICY))
            with patch.object(policy.shutil,'disk_usage',return_value=type('Usage',(),{'free':20*1024**3})()):
                result=policy.check_policy(root)
                self.assertEqual(result['external_drive'],'NOT_ATTEMPTED_USER_AUTHORIZED_DEFER')
                self.assertIn('PENDING',result['mac_local'])
            with patch.object(policy.shutil,'disk_usage',return_value=type('Usage',(),{'free':1})()):
                with self.assertRaisesRegex(RuntimeError,'free space'):policy.check_policy(root)

    def test_inventory_includes_checkpoint_predictions_and_log_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['model.pt','state.pt','predictions.npz','labels.npy','run.log','metrics.csv']:(root/name).write_text(name)
            (root/'runtime_cache').mkdir();(root/'runtime_cache/rebuildable').write_text('cache')
            rows,paths=export.inventory({'results/run_fixture':root})
            self.assertEqual(len(rows),6);self.assertEqual(len(paths),6)
            (root/'linked.pt').symlink_to(root/'model.pt')
            with self.assertRaisesRegex(RuntimeError,'nonregular'):export.inventory({'results/run_fixture':root})

    def fixture(self,root,name='results/run_fixture/model.pt',payload=b'fixture parameters'):
        archive=root/'data.tar.gz';manifest=root/'FILES.json'
        with tarfile.open(archive,'w:gz') as tar:
            info=tarfile.TarInfo(name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        manifest.write_text(json.dumps([dict(path=name,size=len(payload),sha256=hashlib.sha256(payload).hexdigest())]))
        return archive,manifest,export.sha(archive)

    def test_roundtrip_reads_real_extracted_targets_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,m,h=self.fixture(root);dest=root/'verified'
            result=unpack.unpack(a,m,dest,h)
            self.assertTrue(result['all_destination_sha256_verified']);self.assertEqual(result['files'],1)
            with self.assertRaisesRegex(RuntimeError,'exists'):unpack.unpack(a,m,dest,h)
            self.assertEqual((dest/'results/run_fixture/model.pt').read_bytes(),b'fixture parameters')

    def test_archive_tamper_stops_before_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,m,h=self.fixture(root);a.write_bytes(a.read_bytes()+b'tamper')
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):unpack.unpack(a,m,root/'dest',h)
            self.assertFalse((root/'dest').exists())

    def test_manifest_path_escape_and_wrong_file_digest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,m,h=self.fixture(root,name='../escape')
            with self.assertRaisesRegex(RuntimeError,'Unsafe'):unpack.unpack(a,m,root/'dest',h)
            a,m,h=self.fixture(root);rows=json.loads(m.read_text());rows[0]['sha256']='0'*64;m.write_text(json.dumps(rows))
            with self.assertRaisesRegex(RuntimeError,'destination SHA256 mismatch'):unpack.unpack(a,m,root/'dest2',h)

    def test_complete_small_export_archive_and_mac_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            top=Path(tmp);base=top/'results';base.mkdir();run=base/'run_fixture';run.mkdir()
            here=top/'experiments/source';here.mkdir(parents=True)
            (here/'INPUT_REFERENCE.json').write_text(json.dumps({'frozen_source_directories':{}}))
            (here/'mac_unpack_verify.py').write_text('# fixture source')
            (run/'SUCCESS.json').write_text(json.dumps(dict(status='EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE',optimizer_steps=7648,new_test_inference_calls=0)))
            for name in ['run.log','best_checkpoint/model.pt','best_checkpoint/training_state.pt','validation_predictions.npz',
                         'EXP6E3_TRAINING_SUMMARY_CN.md','EXP6E3_CHECKPOINT_RELOAD_VERIFICATION.json','FROZEN_POSTCHECK.json']:
                path=run/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture bytes')
            (run/'exit_code.txt').write_text('0\n')
            with patch.multiple(export,HERE=here,BASE=base,RECORDS=top/'records',MIN_MODEL_BYTES=1):
                out=export.export(run)
            status=json.loads((out/'EXPORT_STATUS.json').read_text())
            self.assertIn('PENDING',status['mac_local']);self.assertEqual(status['github'],'NOT_PUSHED')
            self.assertEqual((base/'LATEST_EXP6E3_EXPORT.txt').read_text().strip(),str(out))
            verified=unpack.unpack(out/'Exp6E3_results.tar.gz',out/'FILES.json',top/'local',status['archive_sha256'])
            self.assertTrue(verified['all_destination_sha256_verified'])
            self.assertEqual((top/'local/results/run_fixture/best_checkpoint/training_state.pt').read_bytes(),b'fixture bytes')

    def test_scientific_executables_remain_byte_identical_and_old_source_unchanged(self):
        here=Path(__file__).resolve().parent;old=here.parent/'exp6e3_matched_gamma_coordinate'
        before=json.loads((here/'ORIGINAL_E3_SOURCE_REFERENCE.json').read_text())
        self.assertEqual(before,{p.name:export.sha(p) for p in old.iterdir() if p.is_file()})
        for name in ['mmodel.py','mrun.py','mprobe.py','mreport.py']:
            self.assertEqual((here/name).read_bytes(),(old/name).read_bytes())
        launch=(here/'launch.sh').read_text()
        self.assertIn('wait "$tee_pid"',launch)
        self.assertLess(launch.index('wait "$tee_pid"'),launch.index('"$here/export_results.py"'))
        self.assertNotIn('"$here/migrate.py" sync',launch)
        self.assertIn('PENDING_USER_TRANSFER',launch)

if __name__=='__main__':unittest.main(verbosity=2)
