# w2vSenti_sentiment-word-embedding-based-on-W2V
graduation project @ huxi 2017.4.18 python TensorFlow-0.9

## w2vSenti:一种由w2v词向量初始化，通过引入情感信息进行再训练得到的词向量


### intro:
  在利用DL技术处理情感分类任务时，词向量的选取往往比DL分类模型的选取更为重要，此次毕设的目的在于比较glove，SSWE，W2V三种模型在CNN和LSTM模型上的分类效果。
  进一步的，我们实现了我们自己的词向量训练方法，并得到了一种改进的词向量w2vSenti。实验表明，w2vSenti能在一定程度上改善分类结果。
  
  
### outline:
  * 我们选用的数据集为[semeval2017 task4:Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)
  
  * 我们采用CNN和LSTM模型作为分类模型
  
  * 我们比较了[glove](http://pan.baidu.com/s/1qX9uVTE)，[SSWE](http://pan.baidu.com/s/1jIoOFRK)，[W2V](http://pan.baidu.com/s/1bZ5TZg)三种模型:其中glove和SSWE为在twitter文本上训练好的词向量，W2V则是由我们自行收集数据，使用gensim工具包进行训练得到
  
  ```python
  #W2V模型使用以下代码加载
  
  from gensim.models import Word2Vec
  model = Word2Vec.load('Word50_3_27.model') 
  print(model['bad'])#通过vec = model['bad'] 进行访问
  ```
  
  * 我们收集了1600万条带有emoji表情符的数据，并按照emoji表达的含义进行了分类，pos文本和neg文本各占800万，[数据地址](http://pan.baidu.com/s/1nv0TTeL)
  
  * 在w2v词向量的基础上，我们继续利用emoji表情符作为标注信息，继续训练词向量，得到了[w2vSentiv词向量模型](http://pan.baidu.com/s/1nvI4NTv)
    
  ```python
  #w2vSenti模型使用以下代码加载
  
  import cPickle  as pickle
  with open(r'w2vSenti.pkl','r') as file:
    model_w2vSenti = pickle.load(file) #加载完后是一个dict数据结构，key为英文词(字符串)，value为np.array类型的向量
  print(model_w2vSenti['bad'])#通过vec = model_w2vSenti['bad'] 进行访问
  ```
  
  * 以上四个词向量作为输入，利用CNN和LSTM模型进行训练和分类，结果如下：
  
  ![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/table3.PNG)
  
  * 由此我们可以得出结论：
  
    * glove&SSWE作为双通道训练在macroaveraged-recall和F1-average两项指标上取得了最好的结果，其混淆矩阵如下：![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/cnn_gls_plot_classify_report.png)
    
    * w2vSenti单独使用时，acc达到单通道结果中的最高值，但是macroaveraged-recall和F1-average两项指标上相对平庸
    
    * glove&SSWE&w2vSenti作为三通道进行联合训练时，取得所有结果中最高的acc(63.5%),其混淆矩阵如下：![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/cnn_gss_plot_classify_report_test0.595.png)
   
  
### details:
<br>**数据来源**：我们选择了semeval17 task4的数据集进行训练和测试。这个数据集是现今在nlp情感分析领域中最有权威性的数据集之一。这个数据集由6000条标注好的twitter文本组成，每条文本的情感极性为{positive,negtive,neutral}中的一个极性。
  
<br>**评价方法**：本次实验中我们采用了macroaveraged-recall，F1-average和acc三种指标来衡量。macroaveraged recall是近年由`Sebastiani F. An axiomatically derived measure for the evaluation of classification algorithms`提出的评判方法，也是semeval17官方评判算法性能的方法，计算方法为各个类别的recall的平均值。F1 average为各个类别的F1值平均值，acc为分类准确率。
  
<br>**分类模型选择**：CNN是目前在DL领域中最为常见的分类模型之一，LSTM则是更适合于文本处理的序列模型，两者相比各有所长。本次实验中我们对每一个词向量都在两种模型上训练，以证明实验结果的有效性。
  * CNN的超参数众多，CNN起初是为了解决图像的分类问题而提出，最早在2014年由`Kim Y. Convolutional neural networks for sentence classification`中被应用在文本情感分类中，众多文献表明，如果能够熟练对CNN调参，往往能取得非常不错的结果。
  * LSTM相对来说更适合文本这一类的序列数据，且超参数较少，不需要大量的调参测试，在我们的github中有之前我们实现的[使用LSTM模型对中文文本进行分类的实验](https://github.com/chrisHuxi/LSTM-sentence-classification/blob/master/lstm_model.ipynb)中，LSTM经过少量调参即可得到远超过[CNN在同一任务中的结果](https://github.com/chrisHuxi/CNN-for-sentence-classification/blob/master/non-static_CNN_for_hotel.ipynb)。

<br>**词向量模型选择**：为了和我们的w2vSenti词向量进行对比，我们选择了。

<br>**文本预处理**：包括用以分类模型的训练的文本（即来自semeval 2017 task4的6000条标注文本），以及用以训练词向量的文本（即爬取的1600万条数据），其处理过程如下，详细可参见[代码:TODO]()：

>1. 分别提取文本和对应的标签
>2. 将每一条文本中的停用词和标点符号去除，包括`via`和`RT`。
>3. 将剩余的文本做tokenize，ie. 将句子进行分词
>4. 将tokenize后的结果与所选择词向量的key值取交集，删去句子中在词向量key中不存在的词
>5. 将取完交集后的词做成一个词典，并将句子中的词按照词典中的索引排列
>>eg.`9725 10161 3449 13194 5505 11149`这种形式
>6. 将数据进行随机打乱(shuffle)

<br>`注：训练分类模型的文本处理与训练词向量的文本略有不同，主要体现在在词向量文本中去除了长度<=9的句子，具体可参见代码`

<br>**分类模型训练与测试**：
  我们将shuffle过的数据分成8:1:1的比例，即6000条文本中的4800条作为训练数据集（train set），600条作为开发集（valid set），最后600条作为测试集（test set）
  <br>**注**：`在6000条数据中pos:neg:neu大概为3:1:2，所以在我们的测试集中大致比例也是如此，在下面的混淆矩阵中可以通过数量来区分类别`
  
* 我们首先在semeval17 task4的数据集上，使用glove词向量进行CNN和LSTM模型的训练。
  * CNN参数设置及测试结果可参见[CNN-glove代码](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/code/glove/CNN-glove.ipynb)
    * 最后结果acc：58.0±0.5 %，混淆矩阵如下：![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/cnn_glove_plot_classify_report.png)
  * LSTM参数设置及测试结果可参见[LSTM-glove代码](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/code/glove/lstm-glove.ipynb)
    * 最后结果acc：54.0±0.5 %，混淆矩阵如下：![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/lstm_glove_plot_classif_report%200.55.png)
    
* 然后我们使用SSWE词向量进行CNN和LSTM模型的训练。
  * CNN参数设置及测试结果可参见[CNN-SSWE代码](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/code/SSWE/CNN-sswe.ipynb)
    * 最后结果acc：59.5±0.5 %，混淆矩阵如下：![](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/img/cnn_glove_plot_classify_report.png)
  * LSTM参数设置及测试结果可参见[LSTM-SSWE代码](https://github.com/chrisHuxi/w2vSenti_sentiment-word-embedding-based-on-W2V/blob/master/code/SSWE/lstm-sswe.ipynb)
    * 最后结果acc：54.0±0.5 %，混淆矩阵如下：![]()

      
      
      
<br>TODO:code上传并解释
