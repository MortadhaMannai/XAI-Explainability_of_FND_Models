import torch
from torch_geometric.nn import GNNExplainer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Union
import torch_geometric.data

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper
from GNNFakeNews.notebooks.util import build_input_from_subgraph


class GNNModelExplainer:
    def __init__(self, model: GNNModelHelper,
                 sample_data: Union[torch_geometric.data.data.Data, torch_geometric.data.batch.Batch], epochs=200,
                 feat_mask_type='feature'):
        """
        class is a manager for explanation pipeline. When initialized, it will explain the model with sample_data
        Parameters
        ----------
        model: GNNModelHelper,
            the model to be explained
        sample_data:  Union[torch_geometric.data.Data, torch_geometric.data.batch.DataBatch]
            the graph data to be explained
        epochs: int,
            epochs that GNNExplainer should run.
        feat_mask_type: str,
            Denotes the type of feature mask that will be learned. Valid inputs are "feature" (a single feature-level
            mask for all nodes), "individual_feature" (individual feature-level masks for each node), and "scalar"
            (scalar mask for each each node). (default: "feature")
        """
        self.subgraph_with_threshold = None
        self.subgraph = None
        self.adjacency_matrix = None
        # pick the root node since it is the news itself, all leaf nodes are the users who shared this news
        self.node_idx = 0
        self.sample_data = sample_data

        self.gnn_explainer = GNNExplainer(model, epochs=epochs, feat_mask_type=feat_mask_type).to(model.m_args.device)
        if model.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
            self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                                   edge_index=sample_data.edge_index,
                                                                                   adj=sample_data.adj,
                                                                                   mask=sample_data.mask)
        elif model.m_hparams.model_type == GNNModelTypeEnum.BIGCN:
            self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                                   edge_index=sample_data.adj,
                                                                                   # batch=sample_data.batch,
                                                                                   bu_edge_index=sample_data.BU_edge_index,
                                                                                   root_index=sample_data.root_index)
        elif model.m_hparams.model_type == GNNModelTypeEnum.UPFD_GCNFN:
            self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                                   edge_index=sample_data.edge_index,
                                                                                   # batch=sample_data.batch,
                                                                                   num_graphs=sample_data.num_graphs)
        else:
            self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                                   edge_index=sample_data.edge_index)
            # batch=sample_data.batch)

    @staticmethod
    def convert_label_to_text(label: torch.Tensor):
        """
        Parameters
        ----------
        label: torch.Tensor,
            torch tensor with possible values 0 and 1.
        """
        print(label)
        if label.all() == 0:
            return 'Fake'
        else:
            return 'Real'

    def visualize_explaining_graph(self, threshold=None, threshold_method='median', save_as=None):
        """
        visualize the subgraph obtained from the GNNExplainer using the edge mask.
        Parameters
        ----------
        threshold: Union[None, float]
            the threshold value for which edge masks to use when visualizing. helps to visualize better. defaults to
            the median of self.edge_mask
        threshold_method: str
            the method to compute the threshold using self.edge_mask. defaults to 'median', possible values are 'mean',
            'median'
        save_as: str,
            if set, saves the resulting images as pdf files in current directory, defaults to None
        """
        assert threshold_method in ['mean', 'median']

        plt.figure(figsize=(8, 8))

        print(f'y: {self.convert_label_to_text(self.sample_data.y.cpu())}')
        if threshold is None:
            if threshold_method == 'mean':
                threshold = torch.mean(self.edge_mask).cpu()
            else:
                threshold = torch.median(self.edge_mask).cpu()
        print(f'Using the threshold method: {threshold_method}')

        print(f'Removing edges with score less than {threshold} with '
              f'min {torch.min(self.edge_mask.cpu(), axis=-1)} and '
              f'max {torch.max(self.edge_mask.cpu(), axis=-1)}')

        indexes = self.edge_mask > threshold

        print(' ############ Graph before dropping edges according to the edge mask ############')

        # y = torch.Tensor([self.sample_data.y.cpu().numpy()[0] for _ in range(self.sample_data.num_nodes)])
        # y = torch.IntTensor([self.sample_data.y.cpu().numpy()[0]] * self.sample_data.num_nodes)
        # print(y)
        ax0, self.subgraph = self.gnn_explainer.visualize_subgraph(node_idx=self.node_idx,
                                                                   edge_index=self.sample_data.edge_index.cpu(),
                                                                   edge_mask=self.edge_mask.cpu(),
                                                                   node_size=500,
                                                                   # y=y,
                                                                   font_size=10)

        plt.axis('off')
        if save_as is not None:
            plt.savefig(f'plot_images/{save_as}_no_threshold.pdf', bbox_inches='tight')
        plt.show()
        print('#################################################################################')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('#################################################################################')
        print(' ############ Graph after dropping edges according to the edge mask ############')
        plt.figure(figsize=(8, 8))
        print(f'Dropping {len(np.where(indexes.cpu().numpy() == False)[0])} edges out of {len(indexes)}')
        num_nodes = self.sample_data.edge_index[:, indexes].clone()
        num_nodes = len(np.unique(np.squeeze(num_nodes.cpu().numpy())))
        print(num_nodes)
        # y = torch.IntTensor([self.sample_data.y.cpu().numpy()[0]] * num_nodes)
        ax, self.subgraph_with_threshold = self.gnn_explainer.visualize_subgraph(node_idx=self.node_idx,
                                                                                 edge_index=self.sample_data.edge_index[
                                                                                            :, indexes]
                                                                                 .cpu(),
                                                                                 edge_mask=self.edge_mask[
                                                                                     indexes].cpu(),
                                                                                 node_size=500,
                                                                                 # y=y,
                                                                                 font_size=10, )

        # node_color=node_colors)
        print(f'Number of nodes before dropping unimportant edges: {self.subgraph.number_of_nodes()}')
        plt.axis('off')
        if save_as is not None:
            plt.savefig(f'plot_images/{save_as}_with_threshold_{threshold_method}.pdf', bbox_inches='tight')
        plt.show()

    def get_node_ids_of_explaining_subgraph(self):
        """
        return the node_ids in self.subgraph
        """
        node_ids = []
        for entry in self.subgraph.nodes.items():
            if entry[0] != 0:
                node_ids.append(entry[0])
        return node_ids

    def visualize_label_dist(self):
        """
        visualize the label distribution of the given batch
        """
        labels_np = self.sample_data.y.cpu().numpy()
        if len(labels_np) == 1:
            print(f'There only one sample with label: {labels_np}')
            return

        unique_labels = np.unique(labels_np)

        fig = plt.figure(figsize=(5, 5))

        ax = fig.add_axes([0, 0, 1, 1])
        for unique_label in unique_labels:
            occurrence_count = len(np.where(labels_np == unique_label)[0])
            ax.bar_label(ax.bar(unique_label, occurrence_count))

        plt.title('Label distribution')
        ax.set_xticks(unique_labels)

        plt.show()

    def visualize_edge_mask_dist(self, highlight_edges=None, save_fig=None):
        """
        scatter plot the edge mask obtained from GNNExplainer
        """
        highlight_edges = highlight_edges if highlight_edges is not None else [[-1, -1]]
        edge_mask_np = self.edge_mask.cpu().numpy()
        sample_data_edge_index = self.sample_data.edge_index.clone().cpu().numpy()
        sg_data = build_input_from_subgraph(self.subgraph, self.sample_data)
        sg_data_edge_index = sg_data.edge_index.clone().cpu().numpy()
        labels = []
        colors = []

        for col_idx in range(sample_data_edge_index.shape[1]):
            current_edge_idx = sample_data_edge_index[:, col_idx]

            # collect edges that are in subgraph
            '''if np.isin(sg_data_edge_index.T, current_edge_idx).all(axis=1).any() and np.isin(highlight_edges,
                                                                                             current_edge_idx).all(
                axis=1).any():
                colors.append('green')
            elif np.isin(sg_data_edge_index.T, current_edge_idx).all(axis=1).any():
                colors.append('orange')
            else:
                colors.append('blue')'''
            colors.append('blue')
            labels.append(f'v_{current_edge_idx[0]},{current_edge_idx[1]}')

        plt.figure(figsize=(20, 50))
        # plt.bar(labels, height=heights, color=colors)
        plt.barh(labels, width=edge_mask_np, color=colors)
        plt.title('Edge mask distribution')
        # plt.xticks(rotation=90)
        if save_fig is not None:
            plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
        plt.show()
        return colors

    def visualize_edge_mask_for_subgraph(self, highlight_edges=None, save_fig=None):
        highlight_edges = highlight_edges if highlight_edges is not None else [[-1, -1]]
        edge_mask_np = self.edge_mask.cpu().numpy()
        sample_data_edge_index = self.sample_data.edge_index.clone().cpu().numpy()
        sg_data = build_input_from_subgraph(self.subgraph, self.sample_data)
        sg_data_edge_index = sg_data.edge_index.clone().cpu().numpy()
        labels = []
        heights = []
        colors = []

        for col_index in range(sg_data_edge_index.shape[1]):
            current_edge_index = sg_data_edge_index[:, col_index]
            if np.isin(highlight_edges, current_edge_index).all(axis=1).any():
                colors.append('green')
            else:
                colors.append('blue')
            # look for the position of the same edge index
            position = (sample_data_edge_index.T == current_edge_index).all(axis=1)
            edge_mask_value = edge_mask_np[position]
            labels.append(f'v_{current_edge_index[0]},{current_edge_index[1]}')
            heights.append(edge_mask_value[0])
        plt.figure(figsize=(20, 8))
        plt.bar(labels, height=heights, color=colors)

        plt.title('Edge mask distribution')
        plt.xticks(rotation=90)
        if save_fig is not None:
            plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
        plt.show()

    def visualize_edge_mask_for_subgraph_with_threshold(self, highlight_edges=None, save_fig=None):
        highlight_edges = highlight_edges if highlight_edges is not None else [[-1, -1]]
        edge_mask_np = self.edge_mask.cpu().numpy()
        sample_data_edge_index = self.sample_data.edge_index.clone().cpu().numpy()
        sg_data = build_input_from_subgraph(self.subgraph_with_threshold, self.sample_data)
        sg_data_edge_index = sg_data.edge_index.clone().cpu().numpy()
        labels = []
        heights = []
        colors = []

        for col_index in range(sg_data_edge_index.shape[1]):
            current_edge_index = sg_data_edge_index[:, col_index]
            if np.isin(highlight_edges, current_edge_index).all(axis=1).any():
                colors.append('green')
            else:
                colors.append('blue')
            # look for the position of the same edge index
            position = (sample_data_edge_index.T == current_edge_index).all(axis=1)
            edge_mask_value = edge_mask_np[position]
            labels.append(f'v_{current_edge_index[0]},{current_edge_index[1]}')
            heights.append(edge_mask_value[0])
        plt.figure(figsize=(20, 8))
        plt.bar(labels, height=heights, color=colors)

        plt.title('Edge mask distribution')
        plt.xticks(rotation=90)
        if save_fig is not None:
            plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
        plt.show()

    def visualize_adjacency_matrix(self):
        """
        show a grayscale image representing the adjacency matrix
        """
        self.adjacency_matrix = nx.to_pandas_adjacency(self.subgraph)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Full adjacency
        ax.set_title('Full Adjacency mask')
        ax.imshow(self.adjacency_matrix, cmap='gray')
        plt.show()
