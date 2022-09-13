## SentenceClassification Model implemented with PyTorch
### `BertForSequenceClassification`代码使用流程


**解压数据集**
`unzip data.zip`

**进入`Bert`代码目录**
`cd BertForSequenceClassification`

**安装依赖**
`pip install -r requirements.txt`

**预处理**
`python preprocess.py`
默认针对SST数据集

**运行训练+测试**

`python main.py`
运行完毕之后，即可得到测试集的效果。

**其他**

`config.yaml`中可以修改相应配置

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
