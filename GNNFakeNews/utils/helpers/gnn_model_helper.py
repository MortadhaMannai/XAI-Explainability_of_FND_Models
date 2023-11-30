from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.data

from GNNFakeNews.utils.eval_helper import *
from GNNFakeNews.utils.enums import GNNModelTypeEnum

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GNNModelHelper(torch.nn.Module):
    """
    helper class for GNN deprecated. reduces code repetition
    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose=True):
        super(GNNModelHelper, self).__init__()
        self.m_args = model_args
        self.m_hparams = model_hparams
        self.m_dataset_manager = model_dataset_manager
        self.verbose = verbose
        self.last_layer = None
        self.last_conv_layer = None
        self.last_layers = {
            'val_y': [],
            'val_last_layer_val': [],
            'train_y': [],
            'train_last_layer_val': [],
            'test_y': [],
            'test_last_layer_val': []
        }
        self.last_conv_layers = {
            'val_y': [],
            'val_last_layer_val': [],
            'train_y': [],
            'train_last_layer_val': [],
            'test_y': [],
            'test_last_layer_val': []
        }

    def get_optimizer(self):
        """
        extension method to handle initialization of optimizer easily
        """
        return torch.optim.Adam(self.parameters(), lr=self.m_hparams.lr, weight_decay=self.m_hparams.weight_decay)

    def m_handle_train(self, data):
        if not self.m_args.multi_gpu:
            data = data.to(self.m_args.device)

        if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
            out, _, _ = self(data.x, data.adj, data.mask)
        elif self.m_hparams.model_type == GNNModelTypeEnum.BIGCN:
            out = self(data.x, data.edge_index, data.batch, data.BU_edge_index, data.root_index)
        elif self.m_hparams.model_type == GNNModelTypeEnum.UPFD_GCNFN:
            out = self(data.x, data.edge_index, data.batch, data.num_graphs)
        else:
            out = self(data.x, data.edge_index, data.batch)

        if self.m_args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y

        return out, y

    @torch.no_grad()
    def compute_test(self, loader_type=None, is_last_epoch=False, loader=None):
        if loader is None:
            loader_types = ['val', 'test']
            if loader_type not in loader_types:
                raise ValueError(f'loader type can only be one of: {loader_types}')
            if loader_type == 'val':
                loader = self.m_dataset_manager.val_loader
            else:
                loader = self.m_dataset_manager.test_loader

        self.eval()
        loss_test = 0.0
        out_log = []
        for data in loader:
            out, y = self.m_handle_train(data)
            if is_last_epoch:
                if loader_type is not None:
                    self.last_layers[f'{loader_type}_y'].append(y)
                    self.last_layers[f'{loader_type}_last_layer_val'].append(self.last_layer)
                    self.last_conv_layers[f'{loader_type}_y'].append(y)
                    self.last_conv_layers[f'{loader_type}_last_layer_val'].append(self.last_conv_layer)
            if self.verbose:
                print(F.softmax(out, dim=1).cpu().numpy())

            out_log.append([F.softmax(out, dim=1), y])

            if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                loss_test += y.size(0) * F.nll_loss(out, y.view(-1)).item()
            else:
                loss_test += F.nll_loss(out, y).item()
        return eval_deep(out_log, loader), loss_test

    def train_then_eval(self):
        """
        extension method to train()
        """
        self.to(self.m_args.device)
        optimizer = self.get_optimizer()
        self.train()
        for epoch in range(self.m_hparams.epochs):
            out_log = []
            loss_train = 0.0
            for data in self.m_dataset_manager.train_loader:
                optimizer.zero_grad()
                out, y = self.m_handle_train(data)
                if epoch == self.m_hparams.epochs - 1:
                    self.last_layers['train_y'].append(y)
                    self.last_layers['train_last_layer_val'].append(self.last_layer)
                    self.last_conv_layers['train_y'].append(y)
                    self.last_conv_layers['train_last_layer_val'].append(self.last_conv_layer)
                if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                    loss = F.nll_loss(out, y.view(-1))
                else:
                    loss = F.nll_loss(out, y)

                loss.backward()
                optimizer.step()

                if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                    loss_train += data.y.size(0) * loss.item()
                else:
                    loss_train += loss.item()

                out_log.append([F.softmax(out, dim=1), y])
            self.m_evaluate_train(out_log, epoch, loss_train)

        return self.m_evaluate_test()

    def m_evaluate_train(self, out_log, epoch, loss_train):
        is_last_epoch = epoch == self.m_hparams.epochs - 1
        acc_train, _, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, self.m_dataset_manager.train_loader)
        [acc_val, _, _, _, _, recall_val, auc_val, _], loss_val = self.compute_test('val', is_last_epoch)

        if self.verbose:
            print(f'\n************** epoch: {epoch} **************'
                  f'\nloss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                  f'\nrecall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
                  f'\nloss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                  f'\nrecall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}'
                  '\n***************************************')

    def m_evaluate_test(self):
        [acc, f1, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = self.compute_test('test',
                                                                                                 is_last_epoch=True)
        if self.verbose:
            print(f'Test set results: acc: {acc:.4f}, f1: {f1}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
                  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
        return acc, precision, recall, f1_macro

    def _m_build_latent_space_values(self, layer: str, split: str):
        if layer == 'convolutional':
            layers = self.last_conv_layers
        else:
            layers = self.last_layers
        if not isinstance(layers[f'{split}_last_layer_val'][0], np.ndarray):
            data = None
            label = None
            for batch_data, batch_label in zip(layers[f'{split}_last_layer_val'], layers[f'{split}_y']):
                if data is None:
                    data = batch_data.cpu().detach().numpy()
                    label = batch_label.cpu().numpy()
                else:
                    data = np.concatenate([data, batch_data.cpu().detach().numpy()])
                    label = np.concatenate([label, batch_label.cpu().numpy()])
            if layer == 'convolutional':
                self.last_conv_layers[f'{split}_last_layer_val'] = data
                self.last_conv_layers[f'{split}_y'] = label
        else:
            data = self.last_conv_layers[f'{split}_last_layer_val']
            label = self.last_conv_layers[f'{split}_y']
        return data, label

    def m_visualize_tsne(self, layer: str, n_components=2, perplexity=10, init='pca', n_iter=1000, learning_rate='auto',
                         split='train', save_fig=None):
        """
        this method should run after train_then_eval is called.
        Parameters
        ----------
        layer: str,
            which layer to visualize
        n_components: int,
            how many dims to output, defaults to 2
        perplexity: int,
            the perplexity to be used in TSNE. defaults to 10
        init: str, ['random', 'pca']
            the initialization method for TSNE, defaults to 'pca'
        n_iter: int,
            iterations required for TSNE, defaults to 1000
        learning_rate: Union[int, str],
            the learning rate for TSNE, defaults to 'auto'
        split: str,
            which output to use. can be outputs from 'train', 'val' or 'test' split.
        save_fig: str,
            if set to None does not save the figure produced by this run, if set to any string value,
            then the figure will be saved under notebooks/plot_images/<save_fig>.pdf.
        """
        assert init in ['random', 'pca']

        if split != 'all':
            data, label = self._m_build_latent_space_values(layer, split=split)
        else:
            data = None
            label = None
            for s in ['train', 'val', 'test']:
                _d, _l = self._m_build_latent_space_values(layer, split=s)
                if data is None:
                    data = _d
                    label = _l
                else:
                    data = np.concatenate([data, _d])
                    label = np.concatenate([label, _l])

        print(f"{split}_last_layer_val size: {data.shape}")
        print(f"{split}_y size: {label.shape}")
        fig, ax = plt.subplots(figsize=(12, 8))

        last_layer_transformed = TSNE(n_components=n_components, learning_rate=learning_rate, init=init,
                                      random_state=42,
                                      perplexity=perplexity, n_iter=n_iter).fit_transform(X=data, y=label)
        last_layer_transformed_real = last_layer_transformed[np.where(label == 1)[0]]
        last_layer_transformed_fake = last_layer_transformed[np.where(label == 0)[0]]

        ax.scatter(last_layer_transformed_real[:, 0], last_layer_transformed_real[:, 1], c='green', label='Real')
        ax.scatter(last_layer_transformed_fake[:, 0], last_layer_transformed_fake[:, 1], c='red', label='Fake')

        ax.set_title(f'TSNE of the last {layer} layer')
        ax.legend(title='Labels')
        if save_fig is not None:
            plt.savefig(f'plot_images/{save_fig}.pdf', bbox_inches='tight')
        plt.show()

    def m_visualize_tsne_of_last_conv_layer(self, n_components=2, perplexity=10, init='pca', n_iter=1000,
                                            learning_rate='auto', split='train', save_fig=None):
        """
        this method should run after train_then_eval is called.
        Parameters
        ----------
        n_components: int,
            how many dims to output, defaults to 2
        perplexity: int,
            the perplexity to be used in TSNE. defaults to 10
        init: str, ['random', 'pca']
            the initialization method for TSNE, defaults to 'pca'
        n_iter: int,
            iterations required for TSNE, defaults to 1000
        learning_rate: Union[int, str],
            the learning rate for TSNE, defaults to 'auto'
        split: str,
            which output to use. can be outputs from 'train', 'val' or 'test' split.
        save_fig: str,
            if set to None does not save the figure produced by this run, if set to any string value,
            then the figure will be saved under notebooks/plot_images/<save_fig>.pdf.
        """
        self.m_visualize_tsne('convolutional', n_components=n_components, learning_rate=learning_rate,
                              perplexity=perplexity, init=init, n_iter=n_iter, split=split, save_fig=save_fig)

    def m_visualize_tsne_of_last_layer_before_classification(self, n_components=2, perplexity=10,
                                                             init='pca', n_iter=1000, learning_rate='auto',
                                                             split='train', save_fig=None):
        """
        this method should run after train_then_eval is called.
        Parameters
        ----------
        n_components: int,
            how many dims to output, defaults to 2
        perplexity: int,
            the perplexity to be used in TSNE. defaults to 10
        init: str, ['random', 'pca']
            the initialization method for TSNE, defaults to 'pca'
        n_iter: int,
            iterations required for TSNE, defaults to 1000
        learning_rate: Union[int, str],
            the learning rate for TSNE, defaults to 'auto'
        split: str,
            which output to use. can be outputs from 'train', 'val' or 'test' split.
        save_fig: str,
            if set to None does not save the figure produced by this run, if set to any string value,
            then the figure will be saved under notebooks/plot_images/<save_fig>.pdf.
        """
        self.m_visualize_tsne('FC', n_components=n_components, learning_rate=learning_rate, perplexity=perplexity,
                              init=init, n_iter=n_iter, split=split, save_fig=save_fig)

    def m_predict(self, sample):
        # toggle to evaluation mode
        self.eval()
        out, y = self.m_handle_train(sample)
        # collect the probabilties
        probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]

        # check if predicted and actual are the same
        if y.cpu().numpy()[0] == np.argmax(probs):
            print(
                f'Predicted the correct label. : Actual is {y.cpu().numpy()[0]} and predicted {np.argmax(probs)} with probability {probs[y]}')
        else:
            print(
                f'Predicted the wrong label: Actual is {y.cpu().numpy()[0]} and predicted {np.argmax(probs)} with probability {probs[y]}')
        return y, probs
