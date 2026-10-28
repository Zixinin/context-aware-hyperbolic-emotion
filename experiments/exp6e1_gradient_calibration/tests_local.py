"""Independent vector/Gram and evidence-boundary tests; stdlib only."""
import ast
import contextlib
import csv
import importlib.util
import io
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibrate as app
from cal_math import calibrate, module_contributions, sum_norm


def norm(v):return math.sqrt(sum(x*x for x in v))
def cos(a,b):return sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b)) if norm(a)*norm(b)>0 else ''
def add(a,b):return [x+y for x,y in zip(a,b)]


def row(i, a=(1.,2.,.5), old=(.1,-.02,.04), new=(.5,.1,-.03), module='shared_encoder'):
    return dict(cohort='c'+str(i),tag='fresh_initial',module=module,
        bce_norm=norm(a),coordinate_aux_norm=norm(old),geodesic_aux_norm=norm(new),
        bce_coordinate_aux_cosine=cos(a,old),bce_geodesic_aux_cosine=cos(a,new),
        old_new_aux_cosine=cos(old,new),coordinate_total_norm=norm(add(a,old)),geodesic_total_norm=norm(add(a,new)))


class CalibrationTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        """Synthetic sealed evidence, never presented as an actual model run."""
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);here=repo/'experiments/exp6e1_gradient_calibration';here.mkdir(parents=True)
            es=repo/'experiments/exp6e_geodesic_alignment';es.mkdir()
            old=repo/'results/exp6e_geodesic_alignment/run_fixture';old.mkdir(parents=True)
            base=repo/'results/exp6e1_gradient_calibration';base.mkdir()
            historical=repo/'results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL';historical.mkdir(parents=True)
            app.write(es/'FROZEN_SOURCES.json',{})
            app.write(es/'PACKAGE_MANIFEST.json',{'FROZEN_SOURCES.json':app.sha(es/'FROZEN_SOURCES.json')})
            app.write(historical/'SUCCESS.json',{'synthetic':True})
            app.write(historical/'INITIALIZATION_PARITY.json',dict(shared_state_sha256='s',hyperbolic_initial_state_sha256='h'))
            train=repo/'数据集/GoEmotions/processed/train.tsv';train.parent.mkdir(parents=True);train.write_text('synthetic fixture only')
            app.write(old/'PLAN.json',dict(source_sha256=app.read(es/'PACKAGE_MANIFEST.json')))
            app.write(old/'INITIALIZATION_PARITY.json',dict(full_initial_state_sha256='full',shared_state_sha256='s',hyperbolic_initial_state_sha256='h',
                historical_A1_initialization_sha256=app.sha(historical/'INITIALIZATION_PARITY.json'),fresh_pretrained=True,
                no_checkpoint_continuation=True,input_scale=1.,all_initial_forward_tensors_exact=True,constructor_rng_exact=True))
            probe=old/'probes/fresh_initial';probe.mkdir(parents=True)
            proof=dict(state_before='full',state_after='full',optimizer_steps=0,test_forward_calls=0,all_grads_none=True)
            for key in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']:proof[key]=True
            app.write(probe/'NO_UPDATE_PROOF.json',proof)
            cohorts={'c'+str(i):[dict(id='id'+str((i*16+j)%145)) for j in range(16)] for i in range(10)}
            app.write(old/'FIXED_TRAIN_COHORTS.json',dict(cohorts=cohorts))
            app.write(old/'dataset_manifest.json',dict(names27=['label'+str(i) for i in range(27)],splits={'train':dict(
                retained_ids=['id'+str(i) for i in range(30587)],source_tsv_sha256=app.sha(train))}))
            all_rows=[];losses=[]
            for i in range(10):
                group=[row(i),row(i,old=(0,0,0),new=(0,0,0),module='classifier'),
                    row(i,a=(0,0,0),module='hyperbolic_projection'),row(i,a=(0,0,0),module='auxiliary_decoder')]
                for r in group:
                    r.update(bce_connected=int(r['bce_norm']>0),coordinate_aux_connected=int(r['coordinate_aux_norm']>0),geodesic_aux_connected=int(r['geodesic_aux_norm']>0))
                all_rows+=group
                losses.append(dict(cohort='c'+str(i),old_total_gradient_norm=math.sqrt(sum(r['coordinate_total_norm']**2 for r in group)),
                    new_total_gradient_norm=math.sqrt(sum(r['geodesic_total_norm']**2 for r in group))))
            app.table(probe/'MODULE_GRADIENTS.csv',all_rows);app.table(probe/'LOSS_SCALES.csv',losses)
            (old/'exit_code.txt').write_text('3\n')
            reference=dict(geodesic_source_files=app.file_map(es),A1_success_sha256=app.sha(historical/'SUCCESS.json'),train_tsv_sha256=app.sha(train),
                fresh_printed_reference={r['cohort']:{'coordinate_aux_norm':round(r['coordinate_aux_norm'],6),'geodesic_aux_norm':round(r['geodesic_aux_norm'],6)}
                    for r in all_rows if r['module']=='shared_encoder'})
            app.write(here/'INPUT_REFERENCE.json',reference)
            app.write(here/'PACKAGE_MANIFEST.json',{'INPUT_REFERENCE.json':app.sha(here/'INPUT_REFERENCE.json')})
            with patch.multiple(app,REPO=repo,HERE=here,BASE=base,OLD=old,E_SOURCE=es):
                app.write(old/'DIAGNOSTIC_SUCCESS.json',dict(status='PRETRAIN_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED',artifacts_sha256=app.old_artifacts()))
                yield base,old,probe

    def test_all_equal_known_gamma_no_extra_point1(self):
        gamma,rows=calibrate([row(i,old=(.2,0,0),new=(1.,0,0)) for i in range(10)])
        self.assertAlmostEqual(gamma,.02,places=15)
        for r in rows:self.assertAlmostEqual(r['matched_to_old_aux_norm'],1.)

    def test_log_median_even_is_geometric_central_pair(self):
        values=[.001,.002,.003,.004,.01,.04,.1,.2,.3,.4]
        gamma,_=calibrate([row(i,old=(v,0,0),new=(.1,0,0)) for i,v in enumerate(values)])
        self.assertAlmostEqual(gamma,math.sqrt(.01*.04),places=15)
        self.assertNotAlmostEqual(gamma,statistics.median(values),places=6)

    def test_permutation_and_no_cohort_drop(self):
        rows=[row(i,old=(.02*(i+1),0,0)) for i in range(10)]
        self.assertEqual(calibrate(rows)[0],calibrate(list(reversed(rows)))[0])
        self.assertEqual(len(calibrate(rows)[1]),10)

    def test_positive_scaling_keeps_direction_including_conflict(self):
        rows=[row(i,a=(1,0,0),new=(-1,1,0)) for i in range(10)]
        gamma,out=calibrate(rows)
        self.assertGreater(gamma,0)
        self.assertTrue(all(r['geodesic_direction_conflicts_with_bce'] for r in out))
        self.assertTrue(all(r['bce_geodesic_aux_cosine']==cos((1,0,0),(-1,1,0)) for r in out))

    def test_cohort_matching_is_not_claimed_exact(self):
        gamma,out=calibrate([row(i,old=(.01*(i+1),0,0)) for i in range(10)])
        ratios=[r['matched_to_old_aux_norm'] for r in out]
        self.assertLess(min(ratios),1);self.assertGreater(max(ratios),1)
        self.assertAlmostEqual(statistics.median([math.log(r) for r in ratios]),0,places=13)

    def test_bce_normalization_cancels_in_matching(self):
        rows=[row(i,a=(100*(i+1),0,0),old=(.2,0,0),new=(1,0,0)) for i in range(10)]
        _,out=calibrate(rows)
        for r in out:self.assertAlmostEqual(r['old_aux_to_bce'],r['matched_aux_to_bce'])

    def test_total_reconstructed_from_independent_vectors(self):
        a,old,new=(1.,-2.,.5),(.3,.1,-.04),(-.7,.1,.2)
        r=row(0,a,old,new)
        result=module_contributions([r],.023)[0]
        self.assertAlmostEqual(result['matched_total_norm'],norm(add(a,[x*.23 for x in new])),places=13)

    def test_perfect_cancellation_and_zero_path(self):
        self.assertEqual(sum_norm(2,2,-1),0)
        self.assertEqual(sum_norm(0,3,None),3)
        r=row(0,a=(0,0,0),module='hyperbolic_projection')
        self.assertAlmostEqual(module_contributions([r],.02)[0]['matched_total_norm'],r['geodesic_aux_norm']*.2)

    def test_classifier_aux_none_keeps_bce(self):
        r=row(0,old=(0,0,0),new=(0,0,0),module='classifier')
        self.assertEqual(module_contributions([r],.02)[0]['matched_total_norm'],r['bce_norm'])

    def test_nonfinite_zero_norm_and_wrong_model_are_rejected(self):
        for field,value in [('bce_norm',0),('coordinate_aux_norm',0),('geodesic_aux_norm',float('nan')),('tag','original_A1_best')]:
            rows=[row(i) for i in range(10)];rows[0][field]=value
            with self.assertRaises(RuntimeError):calibrate(rows)

    def test_missing_duplicate_cosine_and_gram_errors(self):
        with self.assertRaises(RuntimeError):calibrate([row(i) for i in range(9)])
        with self.assertRaises(RuntimeError):calibrate([row(0) for i in range(10)])
        r=row(0);r['bce_geodesic_aux_cosine']=1.1
        with self.assertRaises(RuntimeError):module_contributions([r],.02)
        r=row(0);r['geodesic_total_norm']*=2
        with self.assertRaises(RuntimeError):module_contributions([r],.02)

    def test_manifest_and_path_mutation_rejected(self):
        for name in ['../checkpoint','/absolute','a/../../b']:
            with self.assertRaises(RuntimeError):app.safe_relative(name)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'x.txt').write_text('frozen')
            app.write(root/'PACKAGE_MANIFEST.json',{'x.txt':app.sha(root/'x.txt')})
            with patch.object(app,'HERE',root):
                app.own_source();(root/'x.txt').write_text('changed')
                with self.assertRaises(RuntimeError):app.own_source()

    def test_run_has_no_model_training_imports_or_calls(self):
        source=Path(app.__file__).read_text();tree=ast.parse(source)
        forbidden={'torch','transformers','numpy','subprocess','requests','urllib'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse(any(x.name.split('.')[0] in forbidden for x in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,{'backward','step','load_state_dict','train'})

    def test_report_explicitly_preserves_scope_and_ce_weight_change(self):
        gamma,rows=calibrate([row(i) for i in range(10)])
        result=app.interpretation(gamma,rows,module_contributions([row(i) for i in range(10)],gamma))
        text=app.report(result,rows)
        self.assertIn('辅助CE系数由.1变为',text)
        self.assertIn('s=1保持不变',text)
        self.assertFalse(result['direction_matched']);self.assertFalse(result['pilot_authorized'])

    def test_end_to_end_saved_verification_and_tamper_rejection(self):
        with self.fixture() as (base,old,probe):
            root=base/'run_test';root.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):app.run(root);app.verify(root)
            self.assertTrue((root/'SUCCESS.json').exists())
            proposal=app.read(root/'FROZEN_GAMMA_PROPOSAL.json')
            self.assertFalse(proposal['formal_training_authorized'])
            self.assertEqual(app.read(root/'NO_UPDATE_PROOF.json')['optimizer_steps'],0)
            (root/'RESULT.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Calibration artifacts changed'):app.verify(root)

    def test_input_csv_tamper_or_nontrain_ids_fail_before_calibration(self):
        with self.fixture() as (base,old,probe):
            app.inputs()
            with (probe/'MODULE_GRADIENTS.csv').open('a') as f:f.write('modified')
            with self.assertRaisesRegex(RuntimeError,'artifact seal mismatch'):app.inputs()
        with self.fixture() as (base,old,probe):
            fixed=app.read(old/'FIXED_TRAIN_COHORTS.json');fixed['cohorts']['c0'][0]['id']='not_train'
            app.write(old/'FIXED_TRAIN_COHORTS.json',fixed)
            seal=app.read(old/'DIAGNOSTIC_SUCCESS.json');seal['artifacts_sha256']=app.old_artifacts();app.write(old/'DIAGNOSTIC_SUCCESS.json',seal)
            with self.assertRaises(RuntimeError):app.inputs()


if __name__=='__main__':unittest.main(verbosity=2)
