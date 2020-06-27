# NLP-Space
---

### Papers
---

| Model | Title | Resources | Remarks |
|-------|----------|------------|------|
|word2vec|Efficient Estimation of Word Representations in Vector Space|[[paper]](https://arxiv.org/pdf/1301.3781.pdf) [[pdf]](./papers/Efficient-Estimation-of-Word-Representations-in-Vector-Space.pdf)|------|
|negative sampling|Distributed Representations of Words and Phrases and their Compositionality | [[pdf]](./papers/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)|------|
|Transformer|Attention Is All You Need|[[paper]](https://arxiv.org/abs/1706.03762) [[pdf]](./papers/Attention-is-all-your-need.pdf)|------|

### Learning-Notes
---

[【斯坦福CS224N学习笔记】01-Introduction and Word Vectors](https://zhuanlan.zhihu.com/p/147889351)  
[Word2Vec学习笔记（SVD、原理推导）](https://zhuanlan.zhihu.com/p/148779268)


### NLP Model
---
* [ ] [Word2Vec](./NLP/word2vec)
* [ ] Attention
* [ ] Transformer

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

### Text Matching
---
* [ ] DSSM
* [ ] ESIM
* [ ] DIIN
* [ ] Siamese LSTM

