# 第七届信也科技杯
这是第七届信也科技杯-欺诈用户风险识别的baseline。    
请在比赛网站上下载"初赛数据集.zip"文件，将zip文件中的"phase1_gdata.npz"放到路径'./xydata/raw'中。  
baseline代码中对"phase1_gdata.npz"的train_mask，随机按照6/4的比例将其划分为train/valid dataset。


## Environments
Implementing environment:  
- python = 3.7.6
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- GPU: Tesla V100 32G  


## Training

- **MLP**
```bash
python train.py --model mlp  --epochs 200 --device 0
python inference.py --model mlp --device 0
```

- **GCN**
```bash
python train.py --model gcn  --epochs 200 --device 0
python inference.py --model gcn --device 0
```

- **GraphSAGE**
```bash
python train.py --model sage  --epochs 200 --device 0
python inference.py --model sage --device 0
```

- **GraphSAGE (NeighborSampler)**
```bash
python train_mini_batch.py --model sage_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model sage_neighsampler --device 0
```

- **GAT (NeighborSampler)**
```bash
python train_mini_batch.py --model gat_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model gat_neighsampler --device 0
```

- **GATv2 (NeighborSampler)**
```bash
python train_mini_batch.py --model gatv2_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model gatv2_neighsampler --device 0
```

## 问题分析
​	本题目的预测任务为识别欺诈用户的节点。在数据集中有四类节点，但是预测任务只需要将称为前景节点的欺诈用户（Class 1）从正常用户（Class 0）中区分出来。另外两类称为背景节点的用户（Class 2和 Class 3）尽管在数目上占据更大的比例，但是他们的分类与用户是否欺诈无关。图算法可以通过研究对象之间的复杂关系来提高模型预测效果。而本题提供前景节点和大量的背景节点之间的社交关系，可以充分挖掘各类用户之间的关联和影响力，提出可拓展、高效的图神经网络模型，将隐藏在正常用户中的欺诈用户识别出来。

| 特征名称 | 含义  |
|  :----  | ----  |
| x              | 节点特征，共 17 个                                           |
| y              | 节点共有(0,1,2,3)四类 label，其中测试样本对应的 label 被标为-100 |
| edge_index     | 有向边信息，其中每一行为(id_a, id_b)，代表用户 id_a 指向用户 id_b 的有向边 |
| edge_type      | 边类型                                                       |
| edge_timestamp | 边连接日期其中边日期为从 1 开始的整数，单位为天              |
| train_mask     | 包含训练样本 id 的一维数组                                   |
| test_mask      | 包含测试样本 id 的一维数组 |

# 特征工程

为了提高模型的预测效果，要对题目中给出的数据进行特征工程：

（1）节点特征数据（x），显然该用户本身的数据能够一部分反映出该用户是否为欺诈用户，具有17维。

（2）一个节点的相邻节点，其实际意义是用户的社交信息，出度表示他对别人的信任情况，入度表示别人对他的信任情况。

（3）节点的边的类别信息，用户之间的交易类型显然也是分析欺诈用户的数据。共有11种边，考虑到边的单向性，共22维。

（4）节点的边的连接日期，由于日期数据是连续的，最长时间是578天，将数据按7天进行划分，需要84周。假设节点在某一个时间段中连接节点的频率很高，那么就有可能是欺诈用户。考虑到边的单向性，共有168维。

（5）边的连接数据，为了便与后续模型的处理，将邻接表改为邻接矩阵。

# 模型方法

​	本模型在GraphSAGE模型的基础上结合LightGBM分类器进行欺诈用户预测。

​	GraphSAGE模型对于社交网络中的信息有较好的提取能力。相较于以DeepWalk为代表的Graph Embedding模型无法处理没有看到过的节点，GraphSAGE同时学习每个节点邻域的拓扑结构以及节点特征在领域中的分布，它在关注图中数据的同时也利用了所有图中存在的结构特征，所有GraphSAGE模型也能够应用于没有节点特征的图中。在数据分析中发现节点特征存在大量缺失，而GraphSAGE模型对于此类图的适用度较强。

​	SAGE卷积层以及被集成到torch_geometric.nn库中，无需自己实现。使用以下结构的GraphSAGE模型，edge_index社交网络的基础上，输入17维的节点特征，SAGEConv层之间使用128维进行传递，最后输出2维数据进行判断。

​	使用SAGEConv层处理社交网络信息，输出2维的数据，然后与节点特征、相邻节点特征、相邻边特征、相邻边时间戳特征共同组成217维的数据输入LightGBM模型中，最终输出预测数据。

# 预测结果

| 模型               | AUC      |
| ------------------ | -------- |
| GraphSAGE          | 77.43774 |
| GraphSAGE+LightGBM | 83.16542 |

