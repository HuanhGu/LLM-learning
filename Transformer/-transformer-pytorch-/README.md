# -transformer-pytorch-
基于transformer的机器翻译(pytorch)

- Blog链接： https://www.cnblogs.com/carpell/p/18821064
- 源码链接： https://github.com/fouen6/-transformer-pytorch-/tree/main
- 手搓代码学习链接：https://zhuanlan.zhihu.com/p/1902691586522981311
- Transformer面试题目：https://www.xiaohongshu.com/explore/67d68a4a00000000030289d4?xsec_token=ABmt1EZ_811Krv1irh45Z6nFEiq8LgqKOVTURRo-b-xnM=&xsec_source=pc_like

## 学习问题汇总
1. 词表部分解释tokenizer_XX.json
1️⃣ [UNK] —— Unknown token（未知词）
含义:表示 词表中不存在的词 / 子词

2️⃣ [PAD] —— Padding token（填充符）
含义:用来 把不同长度的句子补齐到同一长度

3️⃣ [SOS] —— Start Of Sequence（序列开始）
含义:表示 序列 / 句子开始

4️⃣ [EOS] —— End Of Sequence（序列结束）
含义:表示 序列 / 句子结束


2. 
- 我不理解我这个代码词表是”中文-italy“,config时预测目标也是“中文-italy”
但是训练模型的时候却下载了“中午英文数据集” 
- 解决：把config的lang_tgt换为"zh"


3. Transformer为什么使用Adam优化器？
https://www.bilibili.com/opus/981776747355701270
随机梯度下降（SGD）在Transformer上的表现明显不如Adam。


4. Feed Forward内容是啥？为什么使用了Feed Forward
FFN, 
ReLU activation

5. 位置编码如何实现？
https://www.bilibili.com/video/BV1AD421g7hs/?spm_id_from=333.337.search-card.all.click&vd_source=5db5a404cbf752d7efa595b25e63b4f9
正余弦波实现

6. Encoder和Decoder有什么区别？




