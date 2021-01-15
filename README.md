## SentenceClassification Model implemented with PyTorch

### 项目简介
本项目复现了用于文本分类的经典模型，目前有TextCNN和BERT，后续会添加LSTM、BiLSTM、FastText等经典模型。
+ **TextCNN**: Kim, Yoon "Convolutional Neural Networks for Sentence Classification." Proceedings of EMNLP. 2014.
**论文地址**：[https://www.aclweb.org/anthology/D14-1181.pdf](https://www.aclweb.org/anthology/D14-1181.pdf)
**详细介绍**：[论文复现TextCNN(基于PyTorch) - 简书](https://www.jianshu.com/p/ed0a82780c20)


### 运行代码
以`TextCNN`为例，介绍代码使用流程, `BertForSequenceClassification`同理

**下载代码**

`git clone git@github.com:unikcc/SentenceClassfication.git`

**进入主项目目录**

`cd SentenceClassfication`

**解压数据集**

`unzip data.zip`

**下载预训练词向量**

打开链接[https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)下载完成后，解压gz文件，并放在主项目目录的`data/embeddings`下；或者其他你想要的位置，在每个项目的config.yaml文件中，embedding_path变量的值可以修改为相应的位置。


**进入TextCNN目录**

`cd TextCNN`
    
**安装依赖**

`pip install -r requirements.txt`

**预处理**

`python preprocess.py`

默认为MR数据集，运行下列命令可以处理SST2数据集
`python preprocess.py --dataset SST2`

**运行训练+测试**

`python main.py`
运行完毕之后，即可得到测试集的效果。

默认为MR数据集，运行下列命令可以处理SST2数据集
`python main.py --dataset SST2`

**修改embededding模式**

`config.yaml`文件中，修改`train_mode`为`random`, `static`, `fine-tuned`即可实现随机初始化、固定词向量和可训练词向量三种模式下的模型训练；

**其他**

`config.yaml`中可以修改相应配置，实现不同数据集的预测，目前支持`SST2`和`MR`，其中`MR`是十折交叉验证；

### 复现结果
**TextCNN**
+ 原始论文

复现结果（括号内为复现结果）
|模型选项|SST-2|MR|
| -- | -- | -- |
| CNN-rand | **82.7**  (80.78) | 76.1  (**77.10**) |
| CNN-static | **86.8**  (85.83) | **81.0**  (80.49) |
| CNN-fine-tuned| **87.2**  (84.68) | **81.5**  (79.88) |
|Bert-base-cased | **93.5** (90.57) | |