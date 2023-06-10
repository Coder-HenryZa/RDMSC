import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import pymysql
import numpy as np


address = {'host': 'localhost', 'user': 'root', 'passwd': 'root', 'database': 'collect_data'}
db = pymysql.connect(host=address.get('host'), user=address.get('user'), passwd=address.get('passwd'),
                     database=address.get('database'))
cursor = db.cursor()

np.set_printoptions(threshold=np.inf)
class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))


def collate_fn(data):
    return data


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        # id 为 eid，index：0,1,2,3,4,5
        id =self.fold_x[index]
        # 读取存储的npz文件
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        # 正向的边矩阵，取边的数量为edge_num*(1-droprate)
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        # 反向的边矩阵，为源矩阵的转置，正向的边矩阵，取边的数量为edge_num*(1-droprate)
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        # new_edgeindex 正向的边矩阵，bunew_edgeindex，反向的边矩阵
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))




class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, dataname, lower=2, upper=100000, droprate=0):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.graph_path = './data/' + dataname + '_Interaction/'
        self.ego_path = './data/' + dataname + '_Ego/'
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(self.graph_path + str(id) + '.npz', allow_pickle=True)
        edgeindex = data['edgeindex']

        row,col = root_edge_enhance(list(edgeindex[0]),list(edgeindex[1]))

        burow = list(col)
        bucol = list(row)
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]


        """
        自我中心网络
        """
        user_ego = np.load(self.ego_path + str(id) + '.npz', allow_pickle=True)
        ego_twitter_id = id
        ego_root_feature = np.array(eval(str(user_ego['root_feature'])))
        ego_tree_feature = np.array(eval(str(user_ego['tree_feature'])))
        ego_edge_index = np.array(eval(str(user_ego['edge_index'])))
        ego_root_index = user_ego['root_index']
        # ego_user_id = user_ego[5]

        """
        end
        """

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])])) , \
               Data(x=torch.tensor(ego_tree_feature, dtype=torch.float32),
                    edge_index=torch.LongTensor(ego_edge_index), y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor([ego_root_feature]),
                    rootindex=torch.LongTensor([int(ego_root_index)]), tree_text_id=torch.LongTensor([int(ego_twitter_id)]))



def root_edge_enhance(row,col):

    # 获取节点序列结合

    c = set(row).union(set(col))

    sorted_list = sorted(c)

    # 获取补充的顶点对

    if sorted_list[0] != 0:
        return row,col

    new_row = []
    new_col = []
    for element in sorted_list[1:]:
        new_row.append(0)
        new_col.append(element)

    # row中找出为0元素的位置
    indices_row = [index for index, value in enumerate(row) if value == 0]

    # 去除a，b中对应索引位置的元素
    row = [row[i] for i in range(len(row)) if i not in indices_row]
    col = [col[i] for i in range(len(col)) if i not in indices_row]

    # col中找出0元素的位置

    indices_col = [index for index, value in enumerate(col) if value == 0]
    # 去除a，b中对应索引位置的元素
    row = [row[i] for i in range(len(row)) if i not in indices_col]
    col = [col[i] for i in range(len(col)) if i not in indices_col]

    row.extend(new_row)
    col.extend(new_col)
    return row,col