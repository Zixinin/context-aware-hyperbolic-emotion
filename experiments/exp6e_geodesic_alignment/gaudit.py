"""Phase1 code/data audit; no proposal assertions and no model/Test inference."""
import csv,hashlib,json,pickle,shutil
from gcommon import *

def source_index(root):
    targets={str(p):p for p in HERE.glob('*.py')}
    for directory in read(HERE/'FROZEN_SOURCES.json'):
        for p in (HERE.parent/directory).glob('*.py'):targets[str(p)]=p
    for name in sc.a1lib.read(sc.a1lib.HERE/'UPSTREAM_REFERENCE.json'):
        if name.endswith(('.py','.tsv')):targets[str(UPSTREAM/name)]=UPSTREAM/name
    index=[]
    for name,p in targets.items():
        text=p.read_text(encoding='utf-8');relative=name.replace(str(REPO)+'/','').replace(str(UPSTREAM)+'/','upstream/')
        destination=root/'source_review'/relative.replace('/','__');destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(p.read_bytes())
        anchors=[dict(line=i,text=line.strip()) for i,line in enumerate(text.splitlines(),1)
                 if line.strip().startswith(('class ','def ')) or any(t in line for t in ['positive_ce =','positive_dot =','auxiliary =','total =','register_buffer(','.expmap0(','.logmap0(','train_data=','label_tree.tsv','reverse=True','maxnorm ='])]
        index.append(dict(path=name,sha256=sha(p),lines=len(text.splitlines()),anchors=anchors,snapshot=str(destination.relative_to(root))))
    write(root/'SOURCE_LINE_INDEX.json',index)
    return index

def labels_audit(root,names):
    reference=read(sc.a1lib.HERE/'UPSTREAM_REFERENCE.json');binpath=UPSTREAM/'label_tree/go_emotion.bin'
    need(sha(binpath)==reference['label_tree/go_emotion.bin'],'Published label binary changed')
    # Deserializing only the already SHA-pinned historical dictionary, never an
    # unknown download; original frozen build also loads this exact binary.
    with binpath.open('rb') as f:vectors=pickle.load(f)
    need(set(vectors)==set(names) and len(vectors)==27 and 'neutral' not in vectors,'Wrong released class set')
    matrix=np.stack([np.asarray(vectors[n],dtype=np.float32) for n in names])
    need(matrix.shape==(27,100) and np.isfinite(matrix).all() and (np.linalg.norm(matrix.astype(np.float64),axis=1)<1).all(),'Invalid fixed label points')
    need(np.array_equal(matrix,np.load(A1/'fixed_emotion_embeddings.npy',allow_pickle=False)),'A1 fixed labels differ from published dictionary')
    original=np.load(sc.a1lib.EXP6B/'original_emotion_embeddings.npy',allow_pickle=False)
    need(sha(sc.a1lib.EXP6B/'original_emotion_embeddings.npy')==read(sc.a1lib.HERE/'EXP6B_REFERENCE.json')['original_emotion_embeddings_sha256'],'Original Exp6B vector anchor changed')
    # Canonical names are identical to upstream IDs in the pinned config; audit
    # their literal mapping without importing the label trainer.
    import ast
    cfg=ast.parse((UPSTREAM/'config.py').read_text())
    maps=[ast.literal_eval(n.value) for n in ast.walk(cfg) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='label2idx' for t in n.targets) and isinstance(n.value,ast.Dict)]
    mapping=next(m for m in maps if set(m)==set(names));need([mapping[n] for n in names]==list(range(27)),'Upstream/canonical order mismatch')
    need(np.array_equal(matrix,original),'Published/Exp6B canonical vector matrices differ')
    tree=UPSTREAM/'data/go_emotion/label_tree.tsv';edges=[tuple(line.split('\t')) for line in tree.read_text().splitlines()]
    need(all(len(e)==2 for e in edges),'Invalid hierarchy records');nodes={n for edge in edges for n in edge}
    need(set(names)<=nodes,'Canonical labels missing from hierarchy')
    train_text=(UPSTREAM/'train_label_embedding.py').read_text()
    required=['label_tree.tsv','Relations(file_path=data_file_path, reverse=True)','train_data=transitive_relations','get_dicts(args.tree, return_emb_dicts=False)']
    need(all(token in train_text for token in required),'Label-training source changed beyond reviewed input route')
    need(not any(token in train_text for token in ['train.csv','test.csv','valid.csv','train.tsv','dev.tsv','test.tsv']),'Label trainer directly refers to example splits; review before training')
    norms=np.linalg.norm(matrix.astype(np.float64),axis=1);np.save(root/'fixed_emotion_embeddings.npy',matrix)
    table(root/'LABEL_VECTOR_AUDIT.csv',[dict(index=i,label=n,dimension=100,norm=float(norms[i]),within_open_ball=True,above_text_FP32_proj_cap=bool(norms[i]>.996),fixed_buffer=True) for i,n in enumerate(names)])
    result=dict(source=str(binpath),source_sha256=sha(binpath),source_commit='97ec6af6b19b831f69cb35a82dc65ea1395fa157',
        shape=[27,100],names=names,fixed=True,label_gradient=False,curvature_magnitude=1.,sectional_curvature=-1.,theoretical_radius=1.,
        text_FP32_proj_cap=.996,minimum_norm=float(norms.min()),maximum_norm=float(norms.max()),outside_ball=0,
        aligned_vector_sha256=sha(root/'fixed_emotion_embeddings.npy'),Exp6B_vector_file=str(sc.a1lib.EXP6B/'original_emotion_embeddings.npy'),
        label_tree=str(tree),label_tree_sha256=sha(tree),tree_nodes=sorted(nodes),nervousment_extra_node='nervousment' in nodes,
        nervousness_canonical_node='nervousness' in nodes,label_pretraining_direct_inputs='fixed hierarchy relations and label vocabulary, no example split paths in reviewed entry point',
        historical_binary_generation_log='INCONCLUSIVE: exact original release training command/log not available',
        leakage_finding='No example-split leakage identified in reviewed data route; complete historical no-leakage certification is unavailable',
        label_construction_during_this_experiment=False,new_label_pretraining=False)
    write(root/'LABEL_AUDIT.json',result);return result

def neutral_audit(root,manifest):
    names=manifest['canonical_names28'];ni=names.index('neutral');rows=[]
    from data27 import OFFICIAL
    for split,(name,_,_) in OFFICIAL.items():
        p=REPO/'数据集/GoEmotions/processed'/name
        with p.open(encoding='utf-8',newline='') as f:raw=list(csv.reader(f,delimiter='\t'))
        sets=[[int(i) for i in r[1].split(',')] for r in raw]
        emotion_counts=np.array([sum(i!=ni for i in labels) for labels in sets]);counts=np.array(list(map(len,sets)))
        actual=dict(split=split,total_28=len(sets),neutral_only=int((emotion_counts==0).sum()),
            neutral_plus_emotion=sum(ni in labels and len(labels)>1 for labels in sets),
            single_positive_28=int((counts==1).sum()),multi_positive_28=int((counts>1).sum()),
            retained_27=int((emotion_counts>0).sum()),single_positive_27=int((emotion_counts==1).sum()),multi_positive_27=int((emotion_counts>1).sum()),
            active_emotion_fraction_full_28=float((emotion_counts>0).mean()),source_sha256=sha(p))
        recorded=manifest['splits'][split]
        need(actual['neutral_only']==recorded['removed_neutral_only'] and actual['neutral_plus_emotion']==recorded['retained_neutral_plus_emotion'] and actual['multi_positive_27']==recorded['multi_emotion'],'Neutral/cardinality audit differs from canonical manifest')
        rows.append(actual)
    table(root/'NEUTRAL_MULTILABEL_COUNTS.csv',rows);return rows

def historical_training(root):
    rows=[]
    for tag,path in [('A0',A0),('A1',A1)]:
        logs=csvrows(path/'batch_log.csv');need(len(logs)==7648,'Historical training budget differs')
        for r in logs:
            for value in r.values():
                if value!='':need(np.isfinite(float(value)),'Nonfinite historical scalar log')
        norms=np.array([float(r['unclipped_grad_norm']) for r in logs])
        rows.append(dict(model=tag,steps=len(logs),preclip_norm_mean=float(norms.mean()),preclip_norm_max=float(norms.max()),
            global_clip_fraction=float((norms>1).mean()),nonfinite_logged_scalars=0,source_sha256=sha(path/'batch_log.csv')))
    table(root/'HISTORICAL_TRAINING_STABILITY.csv',rows);return rows

def report_audits(root,label,neutral,history,index):
    def location(suffix,token):
        row=next(r for r in index if r['path'].endswith(suffix));text=(root/row['snapshot']).read_text().splitlines()
        line=next(i for i,v in enumerate(text,1) if token in v)
        return row['path']+':'+str(line)
    dot=location('/a1model.py','positive_dot =');ce=location('/a1model.py','positive_ce =');buf=location('/a1model.py','register_buffer(')
    md(root/'01_LABEL_EMBEDDING_AUDIT.md',['# 1 标签向量与来源审计','',
        '来源：`'+label['source']+'`；SHA256：`'+label['source_sha256']+'`；固定upstream commit：`'+label['source_commit']+'`。',
        '实际矩阵27×100，标签顺序详见LABEL_VECTOR_AUDIT.csv，neutral不在其中；逐名字核对发布字典、Exp6B矩阵与A1保存矩阵，完全相同。',
        f'范数范围{label["minimum_norm"]:.10f}–{label["maximum_norm"]:.10f}；c=1表示曲率−1，数学有效半径1，超界数0。文本FP32投影保护阈值.996不是数学边界；固定标签超过.996不等于超出球，不重新投影它们。',
        '固定buffer位置：'+buf+'；不在optimizer参数组，不接收梯度。模型构造时核对encoder/decoder/每个超曲层使用相同c=1。',
        '预训练入口读取label_tree.tsv的去root传递关系，get_dicts(...False)提供标签名；没有在所审入口发现读取train/validation/test文本或实例标签的路径。',
        '**来源限制：发布bin原始生成命令、训练日志及taxonomy构造全过程未齐全，不能宣称完整证明历史无泄漏。当前没有发现实例级泄漏证据；如Ubuntu输入/源码核对发现实例划分依赖则停止。**',
        '树同时含正确nervousness及额外nervousment；本轮不合并节点、不修树、不重新预训练，未知其历史数值影响。',
        '所有真实路径、哈希和逐行位置在LABEL_AUDIT.json、SOURCE_LINE_INDEX.json；没有以Proposal替代来源证据。'])
    md(root/'02_AUXILIARY_LOSS_AUDIT.md',['# 2 Forward、Loss与梯度审计','',
        'RoBERTa last_hidden_state[:,0,:]同时进入原27类分类头与HNN。原A1输入尺度1；A1-S才在hyper.encode前乘.03，N1在A1-S基础上加detached距离样本权重。',
        '真正双曲运算：expmap0/proj、Möbius矩阵乘法与加法、logmap0→ReLU→expmap0；辅助decoder先logmap0再普通linear/dropout，输出27个可训练logits，不是固定标签矩阵。',
        '普通坐标运算：分类BCE、decoder线性层、softmax CE，以及z@E.T。其坐标内积不是测地距离，也不是以同一基点定义的切空间黎曼度量inner。',
        '令P_i为全部非neutral正标签，k_i=|P_i|，q_ij=1[j∈P_i]/k_i。C_i=−Σ_j q_ij log softmax(u_i)_j；D_i=Σ_j q_ij z_iᵀe_j。',
        '实际旧辅助loss = (1/B)Σ_i(C_i−D_i)；总loss = mean_{B×27} BCEWithLogits +0.1×辅助loss。两个辅助子项没有独立温度或新增系数。',
        'Soft-target CE位置：'+ce+'；坐标对齐位置：'+dot+'。多正标签均参与，先正标签平均，再完整batch平均；softmax分母仍使标签间存在竞争。',
        '共享梯度路线：aux→decoder/HNN→同一RoBERTa；classifier只接收BCE，固定标签buffer不接收梯度。Ubuntu无更新探针将分别核对BCE/旧aux/新aux的四模块梯度及cosine，失败则不训练。',
        '历史真实训练norm/clip频率在HISTORICAL_TRAINING_STABILITY.csv；没有把仅仅使用球坐标当成全流程均为几何距离。近边界与实际projclip分别记录，数值健康须待真实forward/backward检查。'])
    lines=['# 3 Neutral与多标签审计','',
        '|split|原28总数|neutral-only|neutral+情绪|28单正|28多正|27保留|27单正|27多正|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in neutral:lines.append('|'+r['split']+'|'+'|'.join(str(r[k]) for k in ['total_28','neutral_only','neutral_plus_emotion','single_positive_28','multi_positive_28','retained_27','single_positive_27','multi_positive_27'])+'|')
    lines+=['','真实TSV逐行统计，哈希在CSV；混合neutral确实存在，不能假设不存在。27协议只删除neutral-only，混合样本删除neutral目标后保留全部情绪正标签。',
        '当前A1无单独neutral mask：过滤后每例至少一个情绪，q逐例正标签归一化。其全部batch均参加辅助监督，loss完整batch平均。',
        '旧Exp6B的28类适配：positives=targets[:,:27]，m_i=1[k_i>0]，q_i=positives/max(k_i,1)，neutral-only辅助项为0但分母仍B。因此有效辅助强度随活跃情绪比例Σm_i/B变化。混合neutral+情绪仍接受所有情绪的辅助监督。',
        '未来28标签任务尚未实施；若沿用mask与完整batch平均必须明确报告有效权重，不能未经说明改为仅对active样本平均。A1-G仍严格27类，不创造neutral向量、不运行28类训练。']
    md(root/'03_NEUTRAL_MULTILABEL_AUDIT.md',lines)
    md(root/'04_EUCLIDEAN_CONTROL_AUDIT.md',['# 4 未来matched Euclidean control审计（本轮不训练）','',
        '|因素|可控制程度|建议约束|','|---|---|---|',
        '|数据/encoder/head/seed/预算/阈值|严格控制|同canonical IDs、预训练snapshot、初始化与优化协议|',
        '|维度与参数量|严格控制|100维；与HNN的768×100+bias及100×27+bias一致，保留对应dropout与非线性层数量|',
        '|辅助监督targets|严格控制|全部正标签的同一q、相同soft-target CE与相同归约|',
        '|标签信息来源|可固定来源，几何坐标难完全相同|使用同一标签词表/层级；若logmap0发布球向量则保留来源但改变距离结构，若另训欧氏嵌入则引入嵌入优化差异|',
        '|运算与距离|处理变量|普通linear/ReLU与欧氏距离替代相应几何运算，须明示是距离式还是内积式控制|',
        '|距离/梯度尺度|只能近似匹配|相同系数不代表相同监督强度，先train-only诊断，不按Validation/Test最佳结果匹配尺度|',
        '不能仅增加一个参数量相同的MLP就宣布完全排除了所有非几何差异。固定标签嵌入先验、坐标变换、非线性、距离尺度和梯度裁剪频率都是需披露的因素。',
        'A1-G只检验当前框架内替换对齐目标的效果，不能证明双曲几何优于欧氏几何。没有启动Euclidean Control。'])
    md(root/'05_GEODESIC_DEFINITION_AND_PROTOCOL.md',['# A1-G定义及预登记协议','',
        '距离实现：'+location('/gmath.py','def ball_distance')+'；新辅助项：'+location('/gmath.py','def components')+'；仅新模型实例安装：'+location('/gmodel.py','def install')+'。',
        '保留旧BCE与同一个Soft-target CE Tensor，仅把−mean_positive(zᵀe)替换为+mean_positive d_c(z,e)。',
        'd_c(x,y)=acosh(1+2c||x−y||²/[(1−c||x||²)(1−c||y||²)])/sqrt(c)，c=1。',
        '实现等价式2/sqrt(c)·asinh(sqrt(c)||x−y||/sqrt[(1−c||x||²)(1−c||y||²)])，恒等式cosh(2asinh t)=1+2t²。',
        '球坐标输入仍FP32，距离局部FP64运算保持autograd，再转换为FP32参与正标签平均/辅助batch平均。没有epsilon、边界投影、距离归一化、平方、温度或detach。相同点返回0并使用vector_norm的零次梯度约定，不声称未平方距离在重合点可微。',
        'L_G = BCE + .1·mean_i(C_i + mean_{j∈P_i} d_1(z_i,e_j))。固定标签不接收梯度，新距离须向z及共享encoder回传。',
        '原A1 scale1、seed42、27类、30587/3834/3821、maxlen50、batch16/eval64、4epochs/7648步/warmup765、AdamW5e−5/原WD分组、clip1、FP32/eager、Validation Macro-F1 earliest tie、阈值.30。无新Test、额外seed、欧氏或28类训练。',
        '先与原A1初始化哈希/初始train-eval logits/z/decoder输出/模型参数组核对，CPU重放并核对A1四轮完整ID顺序；A0只核对其真实保存的四个首batch，绝不要求不存在的完整顺序文件。',
        '训练前在fresh及A1 best的相同train-only10cohorts比较旧/新loss与梯度。训练前预登记严重停止门：新total梯度norm或aux绝对尺度超过同forward旧值10倍，或者非有限、球外、错误梯度路径，则停止并报告，不修改权重。训练后的旧坐标目标可能趋近零，epoch探针保留比例但不再以该比例单独停机；非有限/球外/错误梯度仍停止。',
        '即使通过门也完整报告尺度差异和实际clip频率；单独权重敏感性只能在另行定义的实验中进行，不能混入本次单变量对照。'])

def prepare(root):
    import shutil
    need(not (root/'PLAN.json').exists(),'Fresh audit only; no automatic retry')
    source=source_hashes();old_seals();env=runtime();data,manifest=data27()
    snapshot=Path(read(A1/'plan.json')['snapshot'])
    need(sc.a0.snapshot_inventory(snapshot)==read(A1/'plan.json')['snapshot_files_sha256'],'Pretrained snapshot/tokenizer differs')
    need(shutil.disk_usage(BASE).free>=5*1024**3,'Need5GiB free; no cleanup performed')
    shutil.copytree(HERE,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(root/'protected_before.json',protected_inventory(snapshot));write(root/'dataset_manifest.json',manifest);write(root/'config.json',CONFIG)
    index=source_index(root);label=labels_audit(root,manifest['names27']);neutral=neutral_audit(root,manifest);history=historical_training(root)
    report_audits(root,label,neutral,history,index)
    # The original A1 really saved all4 full orders; A0 saved only first batches.
    from torch.utils.data import DataLoader
    ids=[r['id'] for r in data['train']];dl=DataLoader(ids,batch_size=16,shuffle=True,collate_fn=list,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42))
    for epoch in range(1,5):
        order=[rid for b in dl for rid in b]
        need(order==read(A1/f'train_order_epoch{epoch}.json'),'Original A1 full train order not reproducible')
        need(order[:16]==read(A0/f'epoch{epoch}_first_batch.json')['ids'],'Original A0 saved first batch differs')
        write(root/f'planned_train_order_epoch{epoch}.json',order)
    selection=read(sc.REPO/'results/exp6c1s_scaled_train/run_20261022T035554Z_cZrAxW97/FIXED_TRAIN_COHORTS.json')
    need(len(selection['cohorts'])==10 and sum(map(len,selection['cohorts'].values()))==160,'Unexpected fixed train probes')
    train_ids=set(ids);need(all(r['id'] in train_ids for rs in selection['cohorts'].values() for r in rs),'Non-train probe IDs')
    write(root/'FIXED_TRAIN_COHORTS.json',selection)
    write(root/'PLAN.json',dict(created=now(),experiment='Geodesic Alignment Ablation',variant='A1-G',seed=42,
        source_sha256=source,snapshot=str(snapshot),snapshot_sha256=sc.a0.snapshot_inventory(snapshot),runtime=env,
        config_sha256=sha(root/'config.json'),data_sha256=sha(root/'dataset_manifest.json'),new_training_runs=1,new_test_calls=0,
        severe_ratio_gate=SEVERE_RATIO,comparison='Validation seed42 A0/A1/A1-G',blocked_on_smoke=True,
        leakage_evidence_limit=label['historical_binary_generation_log'],no_explicit_example_leakage_found=True))
    freeze_check(root);write(root/'AUDIT_SUCCESS.json',dict(status='PHASE1_CODE_DATA_AUDIT_COMPLETE',historical_provenance_limitation_disclosed=True,
        plan_sha256=sha(root/'PLAN.json'),reports=[f'{i:02d}' for i in range(1,6)],completed=now()))
    print('PHASE1_CODE_DATA_AUDIT_COMPLETE; NO MODEL TRAINING / NO TEST',flush=True)
