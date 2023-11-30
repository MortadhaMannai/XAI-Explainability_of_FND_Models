"""
helper file to handle Bi-GCN model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

import copy as cp

from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from GNNFakeNews.utils.data_loader import *
from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, bu_edge_index, batch, root_index):
        # x, edge_index = data.x, data.BU_edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, bu_edge_index)
        x2 = cp.copy(x)

        # root_index = data.root_index
        # root_index = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_extend = torch.zeros(len(batch), x1.size(1)).to(root_index.device)
        batch_size = max(batch) + 1

        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, bu_edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(batch), x2.size(1)).to(root_index.device)

        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, batch, dim=0)

        return x


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, edge_index, batch, root_index):
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)

        # root_index = data.root_index
        # root_index = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_extend = torch.zeros(len(batch), x1.size(1)).to(root_index.device)
        batch_size = max(batch) + 1

        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(batch), x2.size(1)).to(root_index.device)

        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, batch, dim=0)

        return x


class BiGCNet(GNNModelHelper):
    """

    The Bi-GCN is adopted from the original implementation from the paper authors

    Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
    Link: https://arxiv.org/pdf/2001.06362.pdf
    Source Code: https://github.com/TianBian95/BiGCN

    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose):
        super(BiGCNet, self).__init__(model_args, model_hparams, model_dataset_manager, verbose)
        hid_feats = out_feats = self.m_hparams.n_hidden
        in_feats = self.m_dataset_manager.num_features
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, 2)

    """def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x"""

    def forward(self, x, edge_index, batch, bu_edge_index, root_index):
        TD_x = self.TDrumorGCN(x, edge_index, batch, root_index)
        BU_x = self.BUrumorGCN(x, bu_edge_index, batch, root_index)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def get_optimizer(self):
        """
        custom method for building the optimizer of the model
        """
        BU_params = list(map(id, self.BUrumorGCN.conv1.parameters()))
        if not self.m_args.multi_gpu:
            BU_params += list(map(id, self.BUrumorGCN.conv2.parameters()))
            base_params = filter(lambda p: id(p) not in BU_params, self.parameters())
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': self.BUrumorGCN.conv1.parameters(), 'lr': self.m_hparams.lr / 5},
                {'params': self.BUrumorGCN.conv2.parameters(), 'lr': self.m_hparams.lr / 5}
            ], lr=self.m_hparams.lr, weight_decay=self.m_hparams.weight_decay)
        else:
            BU_params += list(map(id, self.module.BUrumorGCN.conv2.parameters()))
            base_params = filter(lambda p: id(p) not in BU_params, self.parameters())
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': self.module.BUrumorGCN.conv1.parameters(), 'lr': self.m_hparams.lr / 5},
                {'params': self.module.BUrumorGCN.conv2.parameters(), 'lr': self.m_hparams.lr / 5}
            ], lr=self.m_hparams.lr, weight_decay=self.m_hparams.weight_decay)
        return optimizer
