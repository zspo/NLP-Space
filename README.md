# NLP-Space

---

### Papers
| # | Model | Title | Resources | Remarks |
|---|-------|----------|------------|------|
|---|word2vec|Efficient Estimation of Word Representations in Vector Space|[[paper]](https://arxiv.org/pdf/1301.3781.pdf) [[pdf]](./papers/Efficient-Estimation-of-Word-Representations-in-Vector-Space.pdf)|------|
|---|negative sampling|Distributed Representations of Words and Phrases and their Compositionality | [[pdf]](./papers/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)|------|
|---|Transformer|Attention Is All You Need|[[paper]](https://arxiv.org/abs/1706.03762) [[pdf]](./papers/Attention-is-all-your-need.pdf)|------|

### Learning-Notes
| # | Tags | Title | Resources | Remarks |
|---|-------|----------|------------|------|
|---|CS224N|Introduction and Word Vectors|[[note]](./notes/CS224N-2019/CS224N-01-Introduction-and-Word-Vectors.md) [[知乎]](https://zhuanlan.zhihu.com/p/147889351)|------|
|---|word2vec|Word2Vec学习笔记（SVD、原理推导） |[[note]](./notes/Word2Vec学习笔记（CS224N笔记及相关论文学习）.md) [[知乎]](https://zhuanlan.zhihu.com/p/148779268)|------ |

### NLP

* [ ] NLP
    * [ ] Word2Vec  
* [ ] Text Classification
    * [x] Utils
        * [x] generate_w2v: `train word embedding using gensim.`
        * [x] data_helper: `load datasets and data clearning, split to train and valid data.`
    * [x] BaseModel: `a base model, including parameters initialization, embedding initialization, loss function and accuracy, some base api like compile, fit and predict. etc.`
    * [x] FastText
    * [x] TextCNN
    * [x] TextRNN
    * [x] TextBiLSTM
    * [ ] TextRCNN
    * [ ] HAN
    * [ ] BiLSTM+Attention
    * [ ] TransFormer
    * [ ] ...
* [ ] NER
    * [ ] BiLSTM+CRF
* [ ] Text Matching
    * [ ] DSSM
    * [ ] ESIM
    * [ ] DIIN
    * [ ] Siamese LSTM
