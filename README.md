# w2vSenti_sentiment-word-embedding-based-on-W2V
graduation project @ huxi 2017.4.18 python

## w2vSenti:一种由w2v词向量初始化，通过引入情感信息进行再训练得到的词向量

### intro:
  在利用DL技术处理情感分类任务时，词向量的选取往往比DL分类模型的选取更为重要，此次毕设的目的在于比较glove，SSWE，W2V三种模型在CNN和LSTM模型上的分类效果。
  进一步的，我们实现了我们自己的词向量训练方法，并得到了一种改进的词向量w2vSenti。实验表明，w2vSenti能在一定程度上改善分类结果。
  
  
### outline:
  我们选用的数据集为[semeval2017 task4:Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)
  <br>我们采用CNN和LSTM模型作为分类模型
  <br>我们比较了[glove](http://pan.baidu.com/s/1qX9uVTE)，[SSWE](http://pan.baidu.com/s/1jIoOFRK)，[W2V](http://pan.baidu.com/s/1bZ5TZg)三种模型:其中glove和SSWE为在twitter文本上训练好的词向量，W2V则是由我们自行收集数据，使用gensim工具包进行训练得到
  注:glove和SSWE为文本文件，W2V为pkl文件，用cPickle包load即可，load完毕后为一个dict类型
  <br>我们收集了1600万条带有emoji表情符的数据，并按照emoji表达的含义进行了分类，pos文本和neg文本各占800万，数据地址 TODO
  
  <br>TODO:code上传并解释，emoji文本上传
  
      
