***[参考CS224N笔记](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes01-wordvecs1.pdf)
[The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
[word2vec paper](https://arxiv.org/pdf/1301.3781.pdf)
[negative sampling paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)***

@[toc]
### NLP
&#8195;&#8195;人类语言是独特的传达含义的系统，不同于计算机视觉及其他的机器学习任务。
&#8195;&#8195;NLP领域有着不同难度等级的任务，从语音处理到语义解释等。NLP的目标是设计出算法令计算机“理解”自然语言以解决实际的任务。
- Easy的任务包括：拼写纠正、关键词搜索、同义词查找等；
- Medium的任务包括：信息解析等；
- Hard任务包括：机器翻译、情感分析、指代、问答系统等。

### 1、Word Vectors
&#8195;&#8195;英语中估计有13 million单词，他们相互之间并不全是无关的，Feline to cat (猫科动物->猫)、hotel to motel (旅馆->汽车旅馆)等。我们希望用一些向量来编码每个单词，在同一词空间内以点的形式进行表示。直接的方法是构建一个$N(N\le13 million)$维度的空间，这个空间足够将我们的单词进行编码，每个维度可以编码某些我们语言的含义。这些维度可能表示时态、计数、性别等。
&#8195;&#8195;独热编码是直接的编码方法，将每个词表示为$\mathbb{R}^{|V|\times1}$向量，该词在固定顺序下的索引处为1，其他位置都为0。如下
<center><img src="https://img-blog.csdnimg.cn/20200613135818218.png" width=70% /></center>
&#8195;&#8195;每个单词是一个完全独立的个体，如上图所示，结果就是这种词表示方式不能表示任何相似之处，他们都是正交的：

$$(w^{hotel})^Tw^{motel}=(w^{hotel})^Tw^{cat}=0$$

&#8195;&#8195;我们可以尝试将$V$维减小，对原来表示空间进行降维，来寻找一个子空间来进行词关系的表示。
### 2、SVD Based Methods
&#8195;&#8195;奇异值分解方法的做法是，我们首先遍历数据集，通过矩阵$X$存储单词出现的共现次数，然后对$X$进行奇异值分解得出$USV^t$分解。我们可以将$U$的行值可以作为词表中所有词的word embedding。
#### 2.1  Word-Document Matrix
&#8195;&#8195;首先我们可以认为，相关的词总会出现在同一个文档中。譬如，"bank"、"bongs"、"stocks"、"money"等更有可能同时出现，但是"bank"、"octopus"、"banana"等词不可能总是同时出现。我们利用这种共现现象构建一个word-document matrix：X。遍历大量的文档数据集，每当单词$i$和单词$j$同时出现时，我们就在$X_{ij}$位置加1。很明显这将是一个非常大的矩阵($\mathbb{R}^{|V|\times{M}}$)，其中$M$是文档的个数。
#### 2.2 Window based Co-occurrence Matrix（基于窗口的共现矩阵）
&#8195;&#8195;矩阵$X$存储着单词的共现次数。这里我们将在一个合适大小的窗口内来统计单词的共现次数。通过下面的例子进行说明，数据集中有三个句子，窗口大小设定为1：
```
1. I enjoy flying.
2. I like NLP.
3. I like deep learning.
```
根据窗口为1的设定，统计结果矩阵如下：
<center><img src="https://img-blog.csdnimg.cn/20200616100125825.png" width=70% /></center>

#### 2.3  奇异值分解
通过SVD方法得到word embedding的过程如下：
* 构建$|V|\times{|V|}$的共现矩阵，$X$。
* 使用SVD得到，$X=USV^{T}$。
* 选择$U$的前$k$个维度，得到$k$维的词向量。
* $\frac{\sum_{i=1}^{k}\sigma_i}{\sum_{i=1}^{|V|}\sigma_i}$表示前$k$个维度的方差。

我们现在对$X$进行SVD处理：
$X=USV^{T}$
<center><img src="https://img-blog.csdnimg.cn/2020061610175746.png" width=100% /></center>
选择k维奇异值向量进行降维：
<center><img src="https://img-blog.csdnimg.cn/20200616102000454.png" width=100% /></center>

#### 2.4 SVD方法小结
&#8195;&#8195;以上的两种方法（Word-Document Matrix 和 Window based Co-occurrence Matrix）都比传统的编码形式有着跟多的语义信息，但是仍然存在着一些问题：
* 矩阵的维度大小不固定，会随新词的添加而变化，语料库大小也随之变化；
* 矩阵过于稀疏，大部分的单词不会同时出现；
* 矩阵维度太高（$\approx10^6\times{10^6}$）；
* 训练成本太高（$O(mn^2)$）；
* 需要加入一些隐含词（不知道这么理解对不对）来解决词频不均衡的问题。

针对以上的问题有一些解决方法：
* 忽略一些词，例如"the"、"he"、"has"等；
* 窗口动态，即根据文档中单词之间的距离加权计算共现计数；
* 使用皮尔逊相关系数，Use Pearson correlation and set negative counts to 0 instead ofusing just raw count.

### 3、Iteration Based Methods - Word2vec
&#8195;&#8195;我们尝试一种新得方法，通过构建模型能够迭代学习，最终可以根据给定的上下文来对单词的概率进行编码。这个方法设计出的模型的参数就是词向量。在每次的训练迭代过程中，计算误差，更新参数，最终学习出词向量。这个想法可以追溯到1986年，称之为“反向传播（backpropagating）”[[Rumelhart et al., 1988](#refer)]，模型任务越简单，训练速度越快。有一些方法也被尝试过，[[[Collobert et al., 2011](#refer)]构建了NLP模型，第一步是将每个词转为向量，对于每种任务（命名实体识别，词性标注等）不仅训练模型参数同时训练向量，在有不错的效果的同时也得到了好的词向量。
&#8195;&#8195;Word2vec是2013年Mikolov提出的简单有效的方法[[Mikolov et al., 2013](#refer)]（这种方法依赖于语言学中一个非常重要的假设，即分布相似，即相似的词有相似的语境。）Word2vec是一个算法包：
* 算法部分：continuous bag-of-words (CBOW) and skip-gram. CBOW是通过上下文预测中心词，Skip-gram相反，给定中心词预测上下文。
* 模型训练： negative sampling and hierarchical softmax. 负采样是采样出一定比例的负例，层次softmax是通过一种有效的霍夫曼树结构来计算词的概率。

#### 3.1 语言模型（unigrams，bigrams，trigrams等）
<center>"The cat jumped over the puddle."</center>
&#8195;&#8195;以上面的句子为例。

&#8195;&#8195;首先，我们需要构建一个模型来表示一个单词序列的概率。一个好的语言模型会给有效的好句子一个高的概率值，但是句子"stock boil fish is toy"的概率会很低，因为这不是一个正常有意义的句子。用数学来表达，当给定一个有$n$个单词的句子时，其概率为：

$$P(w_1,w_2,...,w_n)$$

我们采用unigrams（一元模型），即每个单词都是独立的，则:
$$P(w_1,w_2,...,w_n)=\prod_{i=1}^{n}P(w_i)$$

&#8195;&#8195;这个表达式有个明显的问题就是，如果有一组句子，虽然他们有着同样的单词，有的句子有意义，有的句子是乱序无意义的，但是他们的概率确实一样的。因为我们的句子都是有序的，一个单词的概率很大程度上和上一个单词有关系。我们需要基于相邻的两个单词的概率来决定句子的概率，即bigrams（二元模型）：
$$P(w_1,w_2,...,w_n)=\prod_{i=2}^{n}P(w_i|w_{i-1})$$

即使这样，我们考虑的也是两两相邻的单词，而不是整个句子。
#### 3.2 Continuous Bag of Words Model (CBOW)
&#8195;&#8195;对于上述的例子，我们通过上下文{"The"、"cat"、"over"、"the"、"puddle"}来预测或生成出中心词"jumped"，这种方式我们成为Continuous Bag of Words Model (CBOW)。
&#8195;&#8195;对于CBOW模型，首先我们设定已知参数，即将输入句子表示为一个one-hot形式的词向量。输入的one-hot向量表示为$x^{(c)}$，输出表示为$y^{(c)}$，CBOW模型只有一个输出，这个$y$为已知的中心词的one-hot向量。对于每个词，我们通过CBOW都可以学习出两个向量，
* $v$：input vector，当词为上下文时
* $u$：output vector，当词为中心词时

首先介绍一些CBOW模型中涉及到的一些参数：
* $w_i$：词表$V$中的第$i$个词
* $\mathcal{V}\in{\mathbb{R}^{n\times{|V|}}}$：input word matrix
* $v_i$：$\mathcal{V}$中的第$i$行，表示的是$w_i$的输入向量
* $\mathcal{U}\in{\mathbb{R}^{|V|\times{n}}}$：output word matrix
* $u_i$：$\mathcal{U}$中的第$i$行，表示的是$w_i$的输出向量

&#8195;&#8195;我们构建两个矩阵$\mathcal{V}\in{\mathbb{R}^{n\times{|V|}}}$和$\mathcal{U}\in{\mathbb{R}^{|V|\times{n}}}$，其中$n$是我们定义的embedding空间的大小。具体的模型构建步骤如下：
1. 首先我们根据窗口大小$m$确定我们的输入one-hot词向量：$(x^{(c-m)},...x^{(c-1)},x^{(c+1)},...,x^{(c+m)}\in{\mathbb{R}^{|V|}})$，中心词为$x^{(c)}$
2. 得到对应的输入word embedding为$(v_{c-m}=\mathcal{Vx^{(c-m)}},v_{c-m+1}=\mathcal{Vx^{(c-m+1)}},...,v_{c+m}=\mathcal{Vx^{(c+m)}}\in{\mathbb{R}^{n}})$
3. 将这些向量平均得到$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}\in{\mathbb{R}^{n}}$
4. 计算出分数向量$z=\mathcal{U}\hat{v}\in{\mathbb{R}^{|V|}}$，点乘计算的是两个向量的相似度，如果两个词比较接近，那么将会有一个较高的分数
5. 通过softmax将分数转为概率值，$\hat{y}=softmax(z)\in{\mathbb{R}^{|V|}}$
6. 我们希望生成的概率$\hat{y}$来匹配真实的概率$y$，即输出的对应的one-hot向量对应真实的单词

如图展示了CBOW模型细节，我们需要学习出两个转换矩阵。
<center><img src="https://img-blog.csdnimg.cn/20200616163354771.png" width=50% /></center>

&#8195;&#8195;我们需要学习出 $\mathcal{V}$ 和 $\mathcal{U}$ 这两个矩阵，首先确定目标函数。当我们试图从某个真实的概率中学习概率时，会考虑用信息论的方法来度量两个分布的距离，我们这里选用交叉熵（cross entropy）$H(\hat{y},y)$来作为目标函数：
$$H(\hat{y},y)=-\sum_{j=1}^{|V|}y_jlog(\hat{y}_j)$$

$y$是一个one-hot向量，简化目标函数为：
$$H(\hat{y},y)=-y_jlog(\hat{y}_j)$$

因此我们优化目标为：
<center><img src="https://img-blog.csdnimg.cn/20200616164500323.png" width=100% /></center>

我们使用随机梯度下降来更新所有相关的词向量 $u_c$ 和 $v_j$。
#### 3.3 Skip-Gram Model
&#8195;&#8195;Skip-gram是给出中心词"jumped"，来预测或生成上下文词 "The", "cat", "over", "the", "puddle"。Skip-gram model大体上和COBW模型相似，不过我们需要将$x$与$y$互换，即这里输入的one-hot向量是一个，输出向量$y$是多个。我们同样定义两个矩阵 $\mathcal{V}$ 和 $\mathcal{U}$，模型构建步骤如下：
1. 首先生成中心词输入向量$x\in{\mathbb{R}^{|V|}}$
2. 得到中心词的embedding词向量 $v_c=\mathcal{V}x\in{\mathbb{R}^n}$
3. 生成分数向量$z=\mathcal{U}v_c$
4. 转为概率值 $\hat{y}=softmax(z)$，$\hat{y}_{c-m},...,\hat{y}_{c-1},\hat{y}_{c+1},...,\hat{y}_{c+m}$是每个上下文词的概率值
5. 目标是让概率分布与真实的接近
<center><img src="https://img-blog.csdnimg.cn/20200616183729454.png" width=50% /></center>
和CBOW模型一样，我们需要确定目标函数，这里我们使用朴素贝叶斯估计来求解出结果。
<center><img src="https://img-blog.csdnimg.cn/20200616183948527.png" width=100% /></center>
利用这个目标函数，我们可以计算出未知参数的梯度，并在每次迭代时通过随机梯度下降来更新它们。
注意到：
<center><img src="https://img-blog.csdnimg.cn/20200616184141501.png" width=60% /></center>

其中 $H(\hat{y},y_{c-m+j})$ 是概率分布向量 $\hat{y}$ 和one-hot向量 $y_{c-m+j}$ 的交叉熵。
#### 3.4 Negative Sampling
&#8195;&#8195;我们注意到目标函数中的 $|V|$ 的值是非常大的，结果就是每次更新或评估目标函数的时候我们都要花费 $O(|V|)$（计算softmax归一化的时候），一个简单的做法就是近似估计它就可以了。
&#8195;&#8195;在每次训练的时候，我们不需要遍历所有的词表，只需要采样少数的负样本。我们基于噪声分布 $P_n(w)$ 采样，其采样概率和词频顺序相匹配。
&#8195;&#8195;Negative Sampling见[paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)。负采样基于Skip-Gram模型，实际上是优化不同的目标。考虑中心词和上下文词对$(w,c)$，如果这个词对来自语料数据集，则概率为$P(D=1|w,c)$，相反，如果词对不是来自语料库的，则为$P(D=0|w,c)$，首先，利用sigmoid函数表示概率值：
$$P(D=1|w,c,\theta)=\sigma(v_c^{T}v_w)=\frac{1}{1+e^{-v_c^{T}v_w}}$$

我们现在构建一个新的目标函数，其目标是maximize两个概率值 $P(D=1|w,c,\theta)$ 和 $P(D=0|w,c,\theta)$，我们利用最大化似然来估计这两个概率分布（这么我们将$\theta$作为模型的参数，在这里表示是 $\mathcal{V}$ 和 $\mathcal{U}$）
<center><img src="https://img-blog.csdnimg.cn/20200616190533719.png" width=100% /></center>

等同于最小化负的对数似然：
<center><img src="https://img-blog.csdnimg.cn/20200616190657796.png" width=100% /></center>

公式中的$\tilde{D}$是负样本集。
对于skip-gram模型，对于给定中心词$c$和上下文词 $c-m+j$表示为：
<center><img src="https://img-blog.csdnimg.cn/20200616191107600.png" width=70% /></center>

对于CBOW模型，中心词为$u_c$，给定的上下文向量为$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}$，目标函数为：
<center><img src="https://img-blog.csdnimg.cn/20200616191307526.png" width=70% /></center>

&#8195;&#8195;现在讨论$P_n(w)$应该是什么。相关大量的讨论似乎是一元模型中的$3/4$次方是最优，为什么是$3/4$，如下：
<center><img src="https://img-blog.csdnimg.cn/20200616191554471.png" width=50% /></center>

"bombastic"的抽样率变成了3倍，但是"is"只是增大了一点点。"is"是不重要的一类词，其出现的概率本来就很大，不需要对其增加很多采样。

#### 3.5 Hierarchical Softmax
&#8195;&#8195;Mikolov同样提出了层次softmax来解决归一化softmax的问题。**在实际中，层次softmax对低频词汇有更好的效果，负采样对高频词和低维向量有着更好的效果。**
<center><img src="https://img-blog.csdnimg.cn/20200616192331651.png" width=70% /></center>

&#8195;&#8195;层次softmax利用二叉树来表示词表中的所有词，树的每个叶子都是一个单词，从根到叶子节点只有唯一的一条路径。每个词没有输出表示，图的每个节点（除了根和叶）都是模型要学习的向量。
&#8195;&#8195;在层次softmax中，单词$w$的向量为$w_i$。$P(w|w_i)$是从根随机游走到叶子节点$w$的概率。最大的优点就是这种计算概率的方式其成本为$O(log(|V|))$，与路径长度相关。
&#8195;&#8195;令$L(w)$为从根到叶子$w$路径上的节点个数，令$n(w,i)$为路径上的第$i$个节点。因此，$n(w,1)$是根节点，$n(w,L(w))$表示的是节点$w$。对于每个节点$n$，我们可以选择其的一个孩子称为$ch(n)$（总是左节点）。我们计算$P(w|w_i)$为：
<center><img src="https://img-blog.csdnimg.cn/20200616193346290.png" width=90% /></center>
其中：
<center><img src="https://img-blog.csdnimg.cn/20200616193421686.png" width=40% /></center>

$\sigma(\cdot)$是sigmoid函数。
&#8195;&#8195;分析上述的公式，首先，我们根据根到叶子节点的路径上各项的乘积。因为我们假设了$ch(n)$总是$n$的左节点，因此当路径游走到左节点时$[n(w,j+1)=ch(n(w,j))]$为1，游走到右边为-1。
&#8195;&#8195;此外，$[n(w,j+1)=ch(n(w,j))]$是一种归一化的方式。对于节点$n$，计算游走到左边的概率和右边的概率，对于每个$v_n^Tv_{w_i}$的概率都是1：
$$\sigma(v_n^Tv_{w_i})+\sigma(-v_n^Tv_{w_i})=1$$

这样确保了$\sum_{w=1}^{|V|}P(w|w_i)=1$，这是原本的softmax。
&#8195;&#8195;最后，我们比较利用点乘来比较输入向量$v_{w_i}$和每个内部的节点向量$v_{n(w,j)}^T$的相似度。对于二叉树图示中的例子来讲，$w_2$，我们需要从根部走两个左边和一个右边达到$w_2$：
<center><img src="https://img-blog.csdnimg.cn/20200616194918228.png" width=100% /></center>

&#8195;&#8195;训练模型的时候，我们目标依然是最小化负对数似然：$-logP(w|w_i)$，但是这里我们不需要更新每个单词的向量，只需要更新该路径上经过的节点的向量即可。
&#8195;&#8195;这种方法的速度取决于二叉树的构造方式和单词分配给叶节点的方式。Mikolovlion利用二叉霍夫曼树，其特点是高频词在树中有着更短的路径。

<div id="refer"></div>

### References
[Rumelhart et al., 1988] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1988).Neurocomputing: Foundations of research. chapter Learning Representations by Back-propagating Errors, pages 696-699. MIT Press, Cambridge, MA, USA.
[Collobert et al., 2011] Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., and Kuksa, P. P. (2011). Natural language processing (almost) from scratch. CoRR, abs/ 1103. 0398.
[Mikolov et al., 2013] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efﬁcient estimation of word representations in vector space. CoRR, abs/ 1301. 3781.

