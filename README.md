# TensorFlow_Space

**这是和自己的笔记文章配套的一系列实践代码。**

**浮沙之上，勿筑高台**

---

### NLP

文本分类、命名实体识别、关系抽取、词性标注、句法分析、文本匹配等等

* [ ] NLP
    * [ ] Word2Vec  
* [ ] 文本分类
    * [x] Utils
        * [x] generate_w2v: 利用gensim训练word2vec，生成需要的词向量
        * [x] data_hepler: 读取原始数据，进行数据清洗预处理等，包括tokenizer编码，pad_sequence，根据word2index产出需要的embedding matrix，生成训练需要的训练集和验证集
    * [x] BaseModel 基本的一个文本分类的类，包括模型参数初始化，embedding初始化，loss、accuracy初始化，包括compile、fit、predict等基础方法（有点儿low，先 凑合用）
    * [x] FastText
    * [x] TextCNN
    * [x] TextRNN
    * [ ] TextRCNN
    * [ ] HAN
    * [ ] LSTM
    * [ ] Bi-LSTM+Attention
    * [ ] TransFormer
* [ ] 命名实体识别
    * [ ] BiLSTM+CRF
* [ ] 文本匹配
    * [ ] DSSM
    * [ ] ESIM
    * [ ] DIIN
    * [ ] Siamese LSTM

### RecSys

推荐系统方向的实践
包括CTR预估模型的一些算法模型实践

* [ ] TFCtr
    * [ ] LR  
    * [ ] GBDT  
    * [ ] GBDT+LR  
    * [ ] FM  
    * [ ] FFM  
    * [ ] DeepFM  
    * [ ] Wide&Deep  
    * [ ] ...

### Basic

* [ ] NN_basic
    * [x] basic tensorflow
    * [x] linear regression
    * [x] logitstic regression
    * [x] simple neural network
    * [x] cnn model
    * [x] rnn lstm model 
    
* [ ] TF_basic  
    * [ ] csv2tfrecord
