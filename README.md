## SentenceClassification Model implemented with PyTorch

### 项目简介
本项目复现了用于文本分类的经典模型，详细介绍见：[论文复现TextCNN(基于PyTorch)](https://www.jianshu.com/p/ed0a82780c20)

目前只有TextCNN一个，后续会添加LSTM、BiLSTM、FastText等经典模型。

### 运行代码

**下载代码**

`git clone git@github.com:unikcc/SentenceClassfication.git`

**进入项目目录**

`cd SentenceClassfication`

**解压数据集**

`unzip data.zip`

**进入项目目录**

`cd TextCNN`
    
**安装依赖**

`pip install requirements.txt`

**预处理**

`python preprocess.py`

**运行训练+测试**

`python main.py`
运行完毕之后，即可得到测试集的效果。

**修改embededding模式**

`config.yaml`文件中，修改`train_mode`为`random`,`static`,`fine-tuned`即可实现随机初始化、固定词向量和可训练词向量三种模式下的模型训练；

**其他**

`config.yaml`中可以修改相应配置，实现不同数据集的预测，目前支持`SST-2`和`MR`，其中`MR`是十折交叉验证；

### 复现结果
+ 原始论文

|模型选项|SST-2|MR|
| -- | -- | -- |
| CNN-rand | 82.7 | 76.1 |
| CNN-static | 86.8 | 81.0 |
| CNN-fine-tuned| 87.2 | 81.5 |
+ 复现结果

|模型选项|SST-2|MR|
| -- | -- | -- |
| CNN-rand | 80.78 |  |
| CNN-static | 85.83 | 80.49  |
| CNN-fine-tuned| 84.68 |   |