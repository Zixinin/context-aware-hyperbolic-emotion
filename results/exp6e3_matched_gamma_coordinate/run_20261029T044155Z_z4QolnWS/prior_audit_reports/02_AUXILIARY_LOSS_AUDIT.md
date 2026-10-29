# 2 Forward、Loss与梯度审计

RoBERTa last_hidden_state[:,0,:]同时进入原27类分类头与HNN。原A1输入尺度1；A1-S才在hyper.encode前乘.03，N1在A1-S基础上加detached距离样本权重。
真正双曲运算：expmap0/proj、Möbius矩阵乘法与加法、logmap0→ReLU→expmap0；辅助decoder先logmap0再普通linear/dropout，输出27个可训练logits，不是固定标签矩阵。
普通坐标运算：分类BCE、decoder线性层、softmax CE，以及z@E.T。其坐标内积不是测地距离，也不是以同一基点定义的切空间黎曼度量inner。
令P_i为全部非neutral正标签，k_i=|P_i|，q_ij=1[j∈P_i]/k_i。C_i=−Σ_j q_ij log softmax(u_i)_j；D_i=Σ_j q_ij z_iᵀe_j。
实际旧辅助loss = (1/B)Σ_i(C_i−D_i)；总loss = mean_{B×27} BCEWithLogits +0.1×辅助loss。两个辅助子项没有独立温度或新增系数。
Soft-target CE位置：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6c1_roberta27_hypaux/a1model.py:26；坐标对齐位置：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6c1_roberta27_hypaux/a1model.py:27。多正标签均参与，先正标签平均，再完整batch平均；softmax分母仍使标签间存在竞争。
共享梯度路线：aux→decoder/HNN→同一RoBERTa；classifier只接收BCE，固定标签buffer不接收梯度。Ubuntu无更新探针将分别核对BCE/旧aux/新aux的四模块梯度及cosine，失败则不训练。
历史真实训练norm/clip频率在HISTORICAL_TRAINING_STABILITY.csv；没有把仅仅使用球坐标当成全流程均为几何距离。近边界与实际projclip分别记录，数值健康须待真实forward/backward检查。
