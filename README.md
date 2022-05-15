# kd-docre
pytorch, doc-level, relation extraction, 

# Intro
this model reimplements **[Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation](https://arxiv.org/pdf/2203.10900v1.pdf)** by pytorch.
if you want to get more detail about this model, plz read this paper.

**Adaptive Focal Loss fuc not reimplement** 

but use **[multilabel_categorical_crossentropy](https://www.spaces.ac.cn/archives/7359)**, you can get more detail from this link


# how to run?

put docred date under dateset/docred/*  dir (don't forget rel_info.json)

then run train.py

if you can't run this model,

plz check your dataset path or reduce batch_size


# result 
dut to not use distant dataset, I don't train the teacher model or student model.

the **best f1 on dev dataset**

**'dev_F1': 58.39, 'dev_F1_ign': 56.24**

the all train hyperparams are default parameters in train.py


# reference 
[Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation](https://arxiv.org/pdf/2203.10900v1.pdf)
[multilabel_categorical_crossentropy](https://www.spaces.ac.cn/archives/7359)
https://github.com/zjunlp/DocuNet
