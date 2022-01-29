![from kaggle](halite-banner.gif)  

# **NLP-Space**
---
* Python3.6.5
* TensorFlow1.14.0
* Pytorch1.8.0
* c++ (for inference)


### Papers
---

| Model | Title | Resources | Remarks |
|-------|----------|------------|------|
|Word2Vec|Efficient Estimation of Word Representations in Vector Space|[[paper]](https://arxiv.org/pdf/1301.3781.pdf)|------|
|negative sampling|Distributed Representations of Words and Phrases and their Compositionality |[[paper]](https://arxiv.org/abs/1310.4546)|------|
|Transformer|Attention Is All You Need|[[paper]](https://arxiv.org/abs/1706.03762)|Google2017|
|Bert|Pre-training of Deep Bidirectional Transformers for Language Understanding|[[paper](https://arxiv.org/abs/1810.04805)]|Google2018|


### Learning-Notes
---

[【斯坦福CS224N学习笔记】01-Introduction and Word Vectors](https://zhuanlan.zhihu.com/p/147889351)  
[Word2Vec学习笔记（SVD、原理推导）](https://zhuanlan.zhihu.com/p/148779268)


### Text Classification
---
* [x] Utils
    * [x] [generate_w2v](./text_classification/utils/generate_w2v.py): train word embedding using gensim.
    * [x] [data_helper](./text_classification/utils/data_helper.py): load datasets and data clearning, split to train and valid data.
* [x] [BaseModel](./text_classification/models/BaseModel.py): a base model, including parameters initialization, embedding initialization, loss function and accuracy, some base api like compile, fit and predict. etc.
* [x] [FastText](./text_classification/models/FastText.py)
* [x] [TextCNN](./text_classification/models/TextCNN.py)
* [x] [TextRNN](./text_classification/models/TextRNN.py)
* [x] [TextBiLSTM](./text_classification/models/TextBiLSTM.py)
* [ ] [TextRCNN](./text_classification/models/TextRCNN.py)
* [ ] HAN
* [ ] BiLSTM+Attention
* [ ] Transformer
* [ ] ...

### NER
---
* [ ] BiLSTM+CRF
* [ ] Bert+CRF
* [ ] Bert+BiLSTM+CRF

### Content Embedding
---
* [x] Bert-Whitening
* [x] Sentence-Bert
* [x] SimCSE
* [ ] ESimCSE

### Text Matching
---
* [ ] Siamese LSTM
* [ ] DSSM
* [x] ESIM
* [ ] DIIN

### Text Generation
---
* [ ] 

### Inference
---
* [ ] onnx (onnxruntime)
* [ ] tensorrt