import torch as th
from torch_scatter import scatter_mean,scatter_max
from torch_geometric.nn import GCNConv
from utils.Attention import Attention
import copy
import torch.nn.functional as F

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class EgoEncoder(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(EgoEncoder, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.droupout_rate = 0.2
        self.w1 = th.nn.Linear(hid_feats * 3, hid_feats * 3)
        self.dropout = th.nn.Dropout(self.droupout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, p=self.droupout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend,x2), 1)

        x = scatter_mean(F.relu(self.w1(x)), data.batch, dim=0)

        return x


class InteractionEncoder(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(InteractionEncoder, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.w1 = th.nn.Linear(hid_feats * 3, hid_feats * 3)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend, x2), 1)

        x = scatter_mean(F.relu(self.w1(x)), data.batch, dim=0)

        return x


class RDMSC(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, atten_out_dim):
        super(RDMSC, self).__init__()
        self.EgoEncoder = EgoEncoder(10, hid_feats, out_feats)
        self.InteractionEncoder = InteractionEncoder(in_feats, hid_feats, out_feats)
        self.Atten = Attention(hid_feats * 3, hid_feats * 3, atten_out_dim)
        self.fc = th.nn.Linear(hid_feats * 6 + atten_out_dim, 4)

    def forward(self, data, data2):
        EE_x = self.EgoEncoder(data2)
        IE_x = self.InteractionEncoder(data)
        query = copy.deepcopy(IE_x.detach())
        key = value = copy.deepcopy(EE_x.detach())
        Attn = self.Atten(query=query, key=key, value=value, dropout=0.5)
        x = th.cat((IE_x, EE_x, Attn), 1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x