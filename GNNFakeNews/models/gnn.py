"""
helper file to handle GNN model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
import torch.nn.functional as F

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper


class GNNet(GNNModelHelper):
    """

    The GCN, GAT, and GraphSAGE implementation

    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose):
        super(GNNet, self).__init__(model_args, model_hparams, model_dataset_manager, verbose)
        num_features = self.m_dataset_manager.num_features
        n_hidden = self.m_hparams.n_hidden
        num_classes = self.m_dataset_manager.num_classes
        model_type = self.m_hparams.model_type

        if model_type == GNNModelTypeEnum.GCN_GNN:
            self.conv1 = GCNConv(num_features, n_hidden)
        elif model_type == GNNModelTypeEnum.SAGE_GNN:
            self.conv1 = SAGEConv(num_features, n_hidden)
        elif model_type == GNNModelTypeEnum.GAT_GNN:
            self.conv1 = GATConv(num_features, n_hidden)
        else:
            raise ValueError(f'Possible Values are {GNNModelTypeEnum.all_elements()}')

        if self.m_hparams.concat:
            self.lin0 = torch.nn.Linear(num_features, n_hidden)
            self.lin1 = torch.nn.Linear(n_hidden * 2, n_hidden)

        self.lin2 = torch.nn.Linear(n_hidden, num_classes)

    """
    def forward(self, data, **kwargs):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_max_pool(x, batch)

        if self.m_hparams.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x
    """

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)
        self.last_conv_layer = h

        # print('After conv layer: ', self.last_conv_layer.data.shape)

        if self.m_hparams.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            # print('news: ', news.data.shape)
            concatenated = torch.cat([news, h], dim=-1)
            # print('Concatenated: ', concatenated.data.shape)
            h = self.lin1(concatenated).relu()
            # print('After lin1: ', h.data.shape)

        self.last_layer = h
        h = self.lin2(h)
        return h.log_softmax(dim=-1)
