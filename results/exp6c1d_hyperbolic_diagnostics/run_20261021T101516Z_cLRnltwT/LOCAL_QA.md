# v1.1 QA

15项本地CPU/合成检查通过，见LOCAL_TESTS.log。其中两项专门复现A0/A1不同产物schema，确认不会要求A0专属不存在文件、也不放宽A1必需证据。其余验证集配对统计/几何/梯度计量/报告/审核包校验保持通过。

没有真实Ubuntu checkpoint梯度或新训练；真实诊断仍待用户执行。旧A0/A1源码、checkpoint、结果不修改。
