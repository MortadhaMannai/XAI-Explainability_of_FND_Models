"""
helper file to handle GNNCL model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

from math import ceil

import torch
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def batch_norm(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'batch_norm{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.batch_norm(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.batch_norm(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.batch_norm(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class GNNCLNet(GNNModelHelper):
    """
     The GNN-CL is implemented using DiffPool as the graph encoder and profile feature as the node feature

     Paper: Graph Neural Networks with Continual Learning for Fake News Detection from Social Media
     Link: https://arxiv.org/pdf/2007.03316.pdf
     """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose):
        super(GNNCLNet, self).__init__(model_args, model_hparams, model_dataset_manager, verbose)

        num_nodes = ceil(0.25 * self.m_hparams.max_nodes)
        in_channels = self.m_dataset_manager.num_features
        num_classes = self.m_dataset_manager.num_classes
        self.gnn1_pool = GNN(in_channels, 64, num_nodes)
        self.gnn1_embed = GNN(in_channels, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.m_hparams.lr)
