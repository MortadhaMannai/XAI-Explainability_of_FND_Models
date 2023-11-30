"""
helper file to handle GCNFN model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv, SAGEConv

from GNNFakeNews.utils.data_loader import *
from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper


class GCNFNet(GNNModelHelper):
    """

    GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder;
    the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional
    comment word2vec (spaCy) embeddings plus 10-dimensional profile features

    Paper: Fake News Detection on Social Media using Geometric Deep Learning
    Link: https://arxiv.org/pdf/1902.06673.pdf


    Model Configurations:

    Vanilla GCNFN: args.concat = False, args.feature = content
    UPFD-GCNFN: args.concat = True, args.feature = spacy

    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose):
        super(GCNFNet, self).__init__(model_args, model_hparams, model_dataset_manager, verbose)

        num_features = self.m_dataset_manager.num_features
        num_classes = self.m_dataset_manager.num_classes

        n_hidden = self.m_hparams.n_hidden

        self.conv1 = GATConv(num_features, n_hidden * 2)
        self.conv2 = GATConv(n_hidden * 2, n_hidden * 2)

        self.fc1 = Linear(n_hidden * 2, n_hidden)

        # if concat is True then the model is UPFD, so we follow the paper of UPFD for best performance.
        if self.m_hparams.concat:
            self.fc0 = Linear(num_features, n_hidden)
            self.fc1 = Linear(n_hidden * 2, n_hidden)

        self.fc2 = Linear(n_hidden, num_classes)

    def forward(self, x, edge_index, batch, num_graphs):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        h = F.selu(self.conv1(x, edge_index))
        h = F.selu(self.conv2(h, edge_index))

        # print(self.conv2.state_dict())

        h = F.selu(global_mean_pool(h, batch))
        self.last_conv_layer = h

        h = F.selu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)

        if self.m_hparams.concat:
            # root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            # root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            # news = x[root]
            news = torch.stack([x[(batch == idx).nonzero().squeeze()[0]] for idx in range(num_graphs)])
            news = F.relu(self.fc0(news))
            h = torch.cat([h, news], dim=1)
            h = F.relu(self.fc1(h))

        self.last_layer = h
        h = F.log_softmax(self.fc2(h), dim=-1)

        return h
