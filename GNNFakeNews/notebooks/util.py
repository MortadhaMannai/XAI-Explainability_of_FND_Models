from typing import Union

import networkx as nx
import torch_geometric.data
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.models import gcnfn, gnn
from GNNFakeNews.models.deprecated import bigcn, gnncl
from GNNFakeNews.utils.helpers.gnn_model_arguments import ModelArguments
from GNNFakeNews.utils.helpers.hyperparameter_factory import HparamFactory
from GNNFakeNews.utils.helpers.gnn_dataset_manager import GNNDatasetManager
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_shutdown_parser, get_args_parser
import os
import unicodedata
import re


# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# import os
# import json


def build_model(model_type: GNNModelTypeEnum, test_mode=False, return_dataset_manager=True, local_load=True,
                hparams=None, verbose=False):
    """
    method is a convenient wrapper to initialize, train then evaluate the model
    Parameters
    ----------
    model_type: GNNModelTypeEnum,
        the model type to be run
    test_mode: bool,
        when set to true, runs 1/5 of the original epochs in the hyperparameter settings
    return_dataset_manager: bool,
        when true, returns the respective instance of GNNDatasetManager with the model
    local_load: bool,
        when true, loads the dataset from local resources, when false downloads the UPFD dataset.
    hparams: HparamFactory,
        the hyperparameters of the model, when None, the default hyperparameters are used.
    verbose: bool,
        when true, outputs more information about the process.

    """
    args = ModelArguments()
    model_hparams = HparamFactory(model_type, test_mode=test_mode) if hparams is None else hparams
    dataset_manager = GNNDatasetManager(local_load=local_load, hparam_manager=model_hparams, multi_gpu=args.multi_gpu)
    if model_type == GNNModelTypeEnum.BIGCN:
        model = bigcn.BiGCNet(model_args=args,
                              model_hparams=model_hparams,
                              model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type in [GNNModelTypeEnum.UPFD_GCNFN, GNNModelTypeEnum.VANILLA_GCNFN]:
        model = gcnfn.GCNFNet(model_args=args,
                              model_hparams=model_hparams,
                              model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type in [GNNModelTypeEnum.GCN_GNN, GNNModelTypeEnum.GAT_GNN, GNNModelTypeEnum.SAGE_GNN]:
        model = gnn.GNNet(model_args=args,
                          model_hparams=model_hparams,
                          model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type == GNNModelTypeEnum.GNNCL:
        model = gnncl.GNNCLNet(model_args=args,
                               model_hparams=model_hparams,
                               model_dataset_manager=dataset_manager, verbose=verbose)
    else:
        raise ValueError(f'Options are {GNNModelTypeEnum.all_elements()}')

    if return_dataset_manager:
        return model, dataset_manager
    return model


def make_n_runs_and_avg_stats(model_type: GNNModelTypeEnum, test_mode=False, return_dataset_manager=True,
                              local_load=True, hparams=None, verbose=False, n=10):
    acc, precision, recall, f1score = 0, 0, 0, 0
    for _ in range(n):
        if return_dataset_manager:
            model, dataset_manager = build_model(model_type=model_type, test_mode=test_mode,
                                                 return_dataset_manager=return_dataset_manager, local_load=local_load,
                                                 hparams=hparams, verbose=verbose)
        else:
            model = build_model(model_type=model_type, test_mode=test_mode,
                                return_dataset_manager=return_dataset_manager, local_load=local_load,
                                hparams=hparams, verbose=verbose)
        a, p, r, f = model.train_then_eval()
        acc += a
        precision += p
        recall += r
        f1score += f
    if return_dataset_manager:
        return model, dataset_manager, acc / n, precision / n, recall / n, f1score / n
    return model, acc / n, precision / n, recall / n, f1score / n


import torch


def visualize_sample(sample: Union[nx.Graph, nx.DiGraph, torch_geometric.data.Data, torch_geometric.data.Batch],
                     save_fig=None):
    """
    visualize a graph sample
    Parameters
    ----------
    sample: Union[nx.Graph, nx.DiGraph, torch_geometric.data.Data, torch_geometric.data.Batch]
        the sample to visualize
    save_fig: Union[None, str],
        if passed a str value, saves the image under that name
    """
    plt.figure(1, (12, 12))
    if isinstance(sample, torch_geometric.data.Data) or isinstance(sample, torch_geometric.data.Batch):
        G = to_networkx(sample, remove_self_loops=True, to_undirected=True)
    else:
        G = sample = remove_self_loops_and_directions(sample)
        sample = from_networkx(sample)
    default_edge_color = ['black'] * sample.edge_index.size(1)
    pos = nx.spring_layout(G, seed=10)
    nx.draw_networkx_nodes(G, pos, node_size=500, cmap='cool')

    nx.draw_networkx_edges(G, pos, edge_color=default_edge_color)

    nx.draw_networkx_labels(G, pos, font_size=10)
    if save_fig is not None:
        plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
    plt.show()


def remove_self_loops_and_directions(G):
    for node in G:
        if G.has_edge(node, node):
            G.remove_edge(node, node)
    return G.to_undirected()


class BertAsAServiceManager:
    def __init__(self, model_dir='models/cased_L-24_H-1024_A-16/', max_seq_len=768):
        self.bert_server = None
        self.bert_client = None
        self._start_bert_server(model_dir, max_seq_len)

    def _start_bert_server(self, model_dir, max_seq_len: int):
        """
        Start a bert server instance and client instance and return them
        Parameters
        ----------
        model_dir: str,
            the name of the model to use. must be downloaded
            from https://github.com/llSourcell/bert-as-service#starting-bertserver-from-python
        max_seq_len: int,
            the maximum input sequence for the Bert Model.
        """
        gnnfakenews_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.listdir()[0])))
        model_dir = os.path.join(gnnfakenews_dir, model_dir)
        args = get_args_parser().parse_args(['-model_dir', f'{model_dir}',
                                             '-max_seq_len', f'{max_seq_len}',
                                             '-show_tokens_to_client',
                                             '-num_worker', '4'
                                             ])

        self.bert_server = BertServer(args)
        self.bert_server.start()
        self.bert_client = BertClient()

    def shutdown_bert_server(self):
        args = get_shutdown_parser().parse_args(['-timeout', '5000', '-ip', 'localhost', '-port', '5555'])
        self.bert_server.shutdown(args)


# here follow the paper and remove special characters like '@' and URLs
def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", text)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ' '.join(new_words)


def remove_special_chars(text):
    # special_chars = ['@', '#', '%', '&', '$', '"', '\'', '<', '>', '|', '-', '_', '/', '{', '}']
    special_chars = ['@']
    return re.sub('@', '', text)


def normalize_text(text):
    # text = remove_non_ascii(text)
    text = remove_URL(text)
    return remove_special_chars(text)


def plot_label_distribution(ax, ds, fake_color, real_color, label):
    fake_news = []
    real_news = []
    for data in ds:
        if data.y.cpu().numpy()[0] == 0:
            # print('Fake news')
            fake_news.append(0)
        else:
            real_news.append(1)
    print(f'len fake news {len(fake_news)}')
    print(f'len real news {len(real_news)}')
    height_offsets = {'fake': 0, 'real': 0}
    for i, rect in enumerate(ax.patches):
        if i % 2 == 0:
            height_offsets['fake'] += rect.get_height()
        else:
            height_offsets['real'] += rect.get_height()
    b = ax.bar(x=['Fake', 'Real'], height=[len(fake_news), len(real_news)],
               bottom=[height_offsets['fake'], height_offsets['real']], color=[fake_color, real_color], label=label)
    ax.bar_label(b, label_type='center', fontsize=20, color='black')


def plot_dataset_label_distribution_by_split(train_ds, val_ds, test_ds, save_fig=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_label_distribution(ax, train_ds, fake_color=CSS4_COLORS.get('dodgerblue'),
                            real_color=CSS4_COLORS.get('dodgerblue'), label='train')
    plot_label_distribution(ax, val_ds, fake_color=CSS4_COLORS.get('darkgreen'),
                            real_color=CSS4_COLORS.get('darkgreen'), label='validation')
    plot_label_distribution(ax, test_ds, fake_color=CSS4_COLORS.get('crimson'), real_color=CSS4_COLORS.get('crimson'),
                            label='test')

    print(f'Total len: {len(train_ds) + len(val_ds) + len(test_ds)}')

    ax.set_xlabel('News Type')
    ax.set_ylabel('Count')
    ax.legend()
    plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
    plt.show()


def collect_labels(dataset):
    labels = []
    for data in dataset:
        labels.append('Fake' if data.y[0].cpu().detach().numpy().item() == 0 else 'Real')
    return labels


'''
from networkx.readwrite import json_graph


def save_mask(G, fname, logdir, expdir, fmt='json', suffix=''):
    pth = os.path.join(logdir, expdir, fname + '-filt-' + suffix + '.' + fmt)
    if fmt == 'json':
        dt = json_graph.node_link_data(G)
        with open(pth, 'w') as f:
            json.dump(dt, f)
    elif fmt == 'pdf':
        plt.savefig(pth)
    elif fmt == 'npy':
        np.save(pth, nx.to_numpy_array(G))


def show_adjacency_full(logdir, expdir, mask, ax=None):
    adj = np.load(os.path.join(logdir, expdir, mask), allow_pickle=True)
    if ax is None:
        plt.figure()
        plt.imshow(adj);
    else:
        ax.imshow(adj)
    return adj


def read_adjacency_full(logdir, expdir, mask, ax=None):
    adj = np.load(os.path.join(logdir, expdir, mask), allow_pickle=True)
    return adj


@interact
def filter_adj(mask, thresh=0.5):
    filt_adj = read_adjacency_full(mask)
    filt_adj[filt_adj < thresh] = 0
    return filt_adj


# EDIT THIS INDEX
MASK_IDX = 0


# EDIT THIS INDEX

# m = masks[MASK_IDX]
# adj = read_adjacency_full(m)


@interact(thresh=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01))
def plot_interactive(m, thresh=0.5):
    filt_adj = read_adjacency_full(m)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.title(str(m));

    # Full adjacency
    ax1.set_title('Full Adjacency mask')
    adj = show_adjacency_full(m, ax=ax1);

    # Filtered adjacency
    filt_adj[filt_adj < thresh] = 0
    ax2.set_title('Filtered Adjacency mask');
    ax2.imshow(filt_adj);

    # Plot subgraph
    ax3.set_title("Subgraph")
    G_ = nx.from_numpy_array(adj)
    G = nx.from_numpy_array(filt_adj)
    G.remove_nodes_from(list(nx.isolates(G)))
    nx.draw(G, ax=ax3)
    save_mask(G, fname=m, fmt='json')

    print("Removed {} edges -- K = {} remain.".format(G_.number_of_edges() - G.number_of_edges(), G.number_of_edges()))
    print("Removed {} nodes -- K = {} remain.".format(G_.number_of_nodes() - G.number_of_nodes(), G.number_of_nodes()))'''


def build_input_from_subgraph(subgraph, sample):
    sg_tensor = from_networkx(subgraph.to_undirected())
    nodes_outgoing = sg_tensor.node_stores[0]['edge_index'][0]
    nodes_incoming = sg_tensor.node_stores[0]['edge_index'][1]
    nodes_in_subgraph = []
    for n_o, n_i in zip(nodes_outgoing, nodes_incoming):
        if n_o.cpu().numpy().sum() not in nodes_in_subgraph:
            nodes_in_subgraph.append(n_o.cpu().numpy().sum())
        elif n_i.cpu().numpy().sum() not in nodes_in_subgraph:
            nodes_in_subgraph.append(n_i.cpu().numpy().sum())
    nodes_in_subgraph.sort()
    node_features = sample.x[nodes_in_subgraph]
    sample = Data(x=node_features, edge_index=sg_tensor.edge_index, y=sample.y,
                  batch=torch.zeros(node_features.size(0), dtype=torch.long))
    return sample


def remove_news_content_from_sample(sample):
    node_features = sample.x.clone()
    batch = sample.batch.clone()
    y = sample.y.clone()
    edge_index = sample.edge_index.clone()
    node_features[0] = torch.zeros_like(node_features[0])
    sample_copy = Data(x=node_features, y=y, batch=batch, edge_index=edge_index)
    return sample_copy


def remove_historical_information(sample):
    node_features = sample.x.clone()
    batch = sample.batch.clone()
    y = sample.y.clone()
    edge_index = sample.edge_index.clone()
    node_features[1:] = torch.zeros_like(node_features[1:])
    sample_copy = Data(x=node_features, y=y, batch=batch, edge_index=edge_index)
    return sample_copy
