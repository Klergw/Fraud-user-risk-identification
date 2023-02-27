# dataset name: XYGraphP1

from utils import XYGraphP1_no_valid
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import pickle

sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.2
              , 'batchnorm': False
              , 'l2':5e-7
             }

gat_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             , 'layer_heads':[4,1]
             }

gatv2_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-6
             , 'layer_heads':[4,1]
             }


@torch.no_grad()
def test(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    #sage_neighsampler新增to_embedding函数
    out = model.inference(data.x, layer_loader, device)
    #out = model.inference(data.x, layer_loader, device)
    #y_pred = out.exp()  # (N,num_classes)
    print(out.size())
    #return y_pred
    return out
            
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    #device = torch.device(device)
    device = torch.device('cpu')
    dataset = XYGraphP1_no_valid(root='./data', name='xydata', transform=T.ToSparseTensor())
    
    nlabels = dataset.num_classes
    if args.dataset =='XYGraphP1': nlabels = 2
        
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
        
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)            
        
    data = data.to(device)
        
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)        
    
    if args.model == 'sage_neighsampler':
        para_dict = sage_neighsampler_parameters
        model_para = sage_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gat_neighsampler':   
        para_dict = gat_neighsampler_parameters
        model_para = gat_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GAT_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gatv2_neighsampler':        
        para_dict = gatv2_neighsampler_parameters
        model_para = gatv2_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GATv2_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')


    model_file = './model_files/{}/{}/model.pt'.format(args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file))

    #data.x embedding
    out = test(layer_loader, model, data, device, no_conv)


    #获取GraphSAGE模型输出的数据
    x_train_all, x_test_all = out[data.train_mask.type(torch.long)], out[data.test_mask.type(torch.long)]
    y_train_all = data.y[data.train_mask.type(torch.long)]
    #读取数据
    items = np.load('data/xydata/raw/data.npz')
    x = items['x']
    y = items['y'].reshape(-1, 1)
    edge_index = items['edge_index']
    edge_type = items['edge_type']
    np.random.seed(42)
    train_mask_t = items['train_mask']
    np.random.shuffle(train_mask_t)
    test_mask = items['test_mask']
    #所有训练和测试数据的标签
    x_train_add = torch.tensor(x[train_mask_t], dtype=torch.float).contiguous()
    x_test_add = torch.tensor(x[test_mask], dtype=torch.float).contiguous()
    set_train_mask_t = set(list(train_mask_t))
    set_test_mask_t = set(list(test_mask))
    #做并集
    set_used = set_train_mask_t | set_test_mask_t
    #边的类别信息
    x_edge = np.zeros([x.shape[0], 22])
    edge_type = edge_type - 1
    #节点的边的时间戳
    x_time = np.zeros([x.shape[0], 168])
    edge_timestamp = items['edge_timestamp']
    edge_timestamp = (edge_timestamp.astype(np.int32) / 7).astype(np.int32)
    #相邻节点的类别信息
    x_pointLabel = np.zeros([x.shape[0], 8])
    #tqdm进度条，特征工程
    for i in tqdm(range(len(edge_index))):
        if edge_index[i][0] in set_used or edge_index[i][1] in set_used:
            x_edge[edge_index[i][0]][edge_type[i]] += 1
            x_edge[edge_index[i][1]][edge_type[i]+10] += 1

            x_time[edge_index[i][0]][:edge_timestamp[i]+1] += 1
            x_time[edge_index[i][1]][84:edge_timestamp[i]+1+84] += 1

            if y[edge_index[i][1]] != -100:
                x_pointLabel[edge_index[i][0]][y[edge_index[i][1]]] += 1
            else:
                x_pointLabel[edge_index[i][0]][:2] += 1

            if y[edge_index[i][0]] != -100:
                x_pointLabel[edge_index[i][1]][y[edge_index[i][0]]+4] += 1
            else:
                x_pointLabel[edge_index[i][1]][4:6] += 1
    #LightGBM的训练、测试数据划分
    train_x_edge = torch.from_numpy(x_edge[train_mask_t])
    test_x_edge = torch.from_numpy(x_edge[test_mask])
    train_x_time = torch.from_numpy(x_time[train_mask_t])
    test_x_time = torch.from_numpy(x_time[test_mask])
    train_x_pointLabel = torch.from_numpy(x_pointLabel[train_mask_t])
    test_x_pointLabel = torch.from_numpy(x_pointLabel[test_mask])
    x_train_all = torch.cat((x_train_add, x_train_all, train_x_edge, train_x_time, train_x_pointLabel), 1)
    x_test_all = torch.cat((x_test_add, x_test_all, test_x_edge, test_x_time, test_x_pointLabel), 1)
    X_train, y_train = x_train_all, y_train_all

    #LightGBM模型训练
    gbm = LGBMClassifier(objective='binary',
                         subsample=0.8,
                         colsample_bytree=0.8,
                         verbosity=0, metric='auc',
                         learning_rate=0.01,
                         n_estimators=1000,
                         min_child_samples=125,
                         max_depth=7,
                         num_leaves=128,
                         reg_alpha=0.1,
                         reg_lambda=0.1,
                         #scale_pos_weight=83.7
                         )

    gbm.fit(X_train, y_train)

    joblib.dump(gbm, 'model_files/LGBM_model.pkl')
    gbm = joblib.load('model_files/LGBM_model.pkl')

    y_pred = gbm.predict_proba(x_test_all, num_iteration=gbm.best_iteration_)
    print(y_pred.shape)
    res = pd.DataFrame({"index": data.test_mask, "label": y_pred[:,1]})
    res.to_csv('./submit/result.csv', encoding='utf-8', index=False)
    np.save("submit/output.npy", y_pred)


if __name__ == "__main__":
    main()
