import re
import types
from platform import python_version
from os import path
from enum import Enum

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, \
    AutoModelForMaskedLM, TrainingArguments, Trainer, pipeline, DataCollatorWithPadding
import shap
from datasets import load_dataset, load_metric, load_from_disk
from Huggingface.noteboooks.deprecated.pipeline_for_hamzab_model import FakeNewsPipelineForHamzaB
from Huggingface.noteboooks.visualization_utils import barplot_first_n_largest_shap_values
import re
from IPython.core.display import HTML, display as ipython_display

print(f'This project is written and tested in Python {python_version()}')
PROJECT_DIR = path.abspath(path.dirname(__file__))
DATA_DIR = path.join(PROJECT_DIR, '../data')


class DatasetTypeEnum(Enum):
    KAGGLE_FAKE_NEWS = {
        'TRUE_NEWS_DIR': 'RobertaFakeNews/True.csv',
        'FAKE_NEWS_DIR': 'RobertaFakeNews/Fake.csv',
        'SOURCE': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download',
        'TEXT_COLNAME': 'text',
        'TITLE_COLNAME': 'title',
    }
    CHINHON_FAKE_TWEET_DETECT = {
        'TRUE_NEWS_DIR': 'Chinhon_FakeTweetDetect/real_50k.csv',
        'FAKE_NEWS_DIR': 'Chinhon_FakeTweetDetect/troll_50k.csv',
        'DIR': 'Chinhon_FakeTweetDetect',
        'SOURCE': 'https://github.com/chuachinhon/transformers_state_trolls_cch',
        'TEXT_COLNAME': 'clean_text',
    }
    # if the dataset is unknown we set it to the most common dataset for now
    UNKNOWN = KAGGLE_FAKE_NEWS


class TransformersModelTypeEnum(Enum):
    CH_FAKE_TWEET_DETECT = {
        'NAME': 'chinhon/fake_tweet_detect',
        'LABEL_MAPPINGS': {'LABEL_0': 'True', 'LABEL_1': 'Fake'},
        'TRAIN_DATASET': DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT,
        'EXTERNAL_LINKS': [
            'https://towardsdatascience.com/detecting-state-backed-twitter-trolls-with-transformers-5d7825945938',
            'https://github.com/chuachinhon/transformers_state_trolls_cch']
    }
    EZ_BERT_BASE_UNCASED_FAKE_NEWS = {
        'NAME': 'elozano/bert-base-cased-fake-news',
        'LABEL_MAPPINGS': {'Fake': 'Fake', 'Real': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.UNKNOWN,
        'EXTERNAL_LINKS': [],
    }
    GA_DISTIL_ROBERTA_BASE_FINETUNED_FAKE_NEWS = {
        'NAME': 'GonzaloA/distilroberta-base-finetuned-fakeNews',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': 'GonzaloA/fake_news',
        'EXTERNAL_LINKS': [],
    }
    GS_ROBERTA_FAKE_NEWS = {
        'NAME': 'ghanashyamvtatti/roberta-fake-news',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.KAGGLE_FAKE_NEWS,
        'EXTERNAL_LINKS': [],
    }
    HB_ROBERTA_FAKE_NEWS = {
        'NAME': 'hamzab/roberta-fake-news-classification',
        'LABEL_MAPPINGS': {'FAKE': 'Fake', 'TRUE': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.KAGGLE_FAKE_NEWS,
        'EXTERNAL_LINKS': []
    }
    JY_FAKE_NEWS_BERT_DETECT = {
        'NAME': 'jy46604790/Fake-News-Bert-Detect',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.UNKNOWN,
        'EXTERNAL_LINKS': [],
    }
    DISTILBERT_VANILLA = {
        'NAME': 'distilroberta-base',
        'TRAIN_DATASET': 'GonzaloA/fake_news',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
    }


class ModelManager:
    """
    class that loads all models from Huggingface and manages various tasks
    """

    def __init__(self, model_type: TransformersModelTypeEnum, device='cpu'):
        # clear cache before starting
        torch.cuda.empty_cache()
        model_name = model_type.value['NAME']
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            self.pipeline = FakeNewsPipelineForHamzaB(model=self.model, tokenizer=self.tokenizer,
                                                      return_all_scores=True,
                                                      device=0)
            self.pipeline.preprocess = types.MethodType(custom_preprocess, self.pipeline)
        elif model_type == TransformersModelTypeEnum.DISTILBERT_VANILLA:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            # self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
            #                                           return_all_scores=True,
            #                                           device=0)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
                                                       return_all_scores=True,
                                                       device=0)
            # force the pipeline preprocess to truncate the outputs for convenience
            self.pipeline.preprocess = types.MethodType(custom_preprocess, self.pipeline)
        self.accuracy_metric = load_metric('accuracy')
        self.precision_metric = load_metric('precision', average='macro')
        self.recall_metric = load_metric('recall', average='macro')
        self.f1_metric = load_metric('f1', average='macro')
        self.trainer = None

    @staticmethod
    def get_training_arguments(logging_steps: int):
        return TrainingArguments(
            output_dir='results',
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            metric_for_best_model='accuracy',
            load_best_model_at_end=False,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_steps=logging_steps,
            push_to_hub=False,
        )

    def train_model(self, train_dataset, val_dataset):
        logging_steps = len(train_dataset) // (2 * 16 * 3)
        self.trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(logging_steps),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        self.trainer.train()

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def compute_test(self, test_dataset):
        predictions = np.argmax(self.trainer.predict(test_dataset).predictions, axis=1)
        self.metric.compute(predictions=predictions, references=test_dataset.label_ids)


class DatasetManager:
    """
    class that handles operations with the dataset GonzaloA/fake_news
    """

    def __init__(self, tokenizer, local_load=True):
        self.tokenizer = tokenizer
        if local_load:
            self.load_local_dataset()
        else:
            dataset = load_dataset('GonzaloA/fake_news')
            self.train_set = dataset.get('train')
            self.val_set = dataset.get('validation')
            self.test_set = dataset.get('test')
            self.train_set_tok = None
            self.val_set_tok = None
            self.test_set_tok = None

    def load_local_dataset(self):
        self.train_set = load_from_disk('dataset_edited/train.hf')
        self.train_set_tok = load_from_disk('dataset_edited/train_tok.hf')  # .shuffle(seed=42)
        self.val_set = load_from_disk('dataset_edited/val.hf')
        self.val_set_tok = load_from_disk('dataset_edited/val_tok.hf')  # .shuffle(seed=42)
        self.test_set = load_from_disk('dataset_edited/test.hf')
        self.test_set_tok = load_from_disk('dataset_edited/test_tok.hf')  # .shuffle(seed=42)

    def tokenize(self, dataset):
        return self.tokenizer(dataset['text'], truncation=True)

    def prepare_dataset(self):
        col_names_to_remove = self.train_set.column_names
        # remove all columns except label
        col_names_to_remove.remove('label')
        self.train_set = self.train_set.map(remove_source_from_news, batched=True)
        self.train_set_tok = self.train_set.map(self.tokenize, batched=True, remove_columns=col_names_to_remove)
        self.val_set = self.val_set.map(remove_source_from_news, batched=True)
        self.val_set_tok = self.val_set.map(self.tokenize, batched=True, remove_columns=col_names_to_remove)
        self.test_set = self.test_set.map(remove_source_from_news, batched=True)
        self.test_set_tok = self.test_set.map(self.tokenize, batched=True, remove_columns=col_names_to_remove)

    def save_dataset(self):
        self.train_set.save_to_disk('dataset_edited/train.hf')
        self.train_set_tok.save_to_disk('dataset_edited/train_tok.hf')
        self.val_set.save_to_disk('dataset_edited/val.hf')
        self.val_set_tok.save_to_disk('dataset_edited/val_tok.hf')
        self.test_set.save_to_disk('dataset_edited/test.hf')
        self.test_set_tok.save_to_disk('dataset_edited/test_tok.hf')


def remove_source_from_news(dataset):
    regex_pattern = r'.*(((\(Reuters\))|(\(REUTERS\))) - )'
    texts = dataset['text']
    texts = [re.sub(regex_pattern, '', text) for text in texts]
    dataset['text'] = texts
    return dataset


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs


def predict_multiple_with_correct_labels(model_manager: ModelManager, texts: list, verbose=False):
    label_mapping = model_manager.model_type.value['LABEL_MAPPINGS']
    raw_predictions = model_manager.pipeline(texts)
    if verbose:
        for i, raw_pred in enumerate(raw_predictions):
            for label_score_map in raw_pred:
                print(f"Sample {i} is predicted {label_mapping[label_score_map['label']]} "
                      f"with score: {label_score_map['score']}")
            print("###################################################################")


class FakeNewsExplainer:
    def __init__(self, model: dict, background: list, explainer='partition'):
        """Class handles all code heavy tasks and returns meaningful data and visualizations for convenience.
        Parameters
        ----------
        model: dict
            a dict with following keys: 'NAME', 'LAB
            EL_MAPPINGS', 'DATASET'
        background: list,
            background data
        explainer: str,
            the algorithm for the explainer, defaults to 'deep' which calls a DeepExplainer for the model.
            if set to any other value, it will call the vanilla explainer of SHAP: shap.Explainer
        """
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.tc_pipeline = None
        self.label_mappings = model['LABEL_MAPPINGS']
        self.dataset = load_dataset(model['DATASET'])
        # self.dataset.cache_files()

        self.load_model(model['NAME'])
        if explainer == 'deep':
            samples = self.get_random_samples(n=200)
            print('Is model supported: ', shap.DeepExplainer.supports_model_with_masker(self.model, self.tokenizer))
            if shap.DeepExplainer.supports_model_with_masker(self.model, self.tokenizer):
                self.explainer = shap.DeepExplainer(self.pipeline, background)
            else:
                self.explainer = shap.Explainer(self.pipeline, masker=shap.maskers.Text(self.tokenizer))

        else:
            self.explainer = shap.Explainer(self.pipeline, masker=shap.maskers.Text(self.tokenizer))

    def load_model(self, model_name: str):
        """load model to the device and create the pipeline that will be used in the explanation
        Parameters
        ----------
        model_name: str
            the model name in huggingface transformers repository
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
                                              return_all_scores=True,
                                              device=0)
        pipeline.preprocess = types.MethodType(custom_preprocess, pipeline)

        self.pipeline = pipeline

        # self.pipeline = shap.models.TransformersPipeline(self.tc_pipeline)

    def compute_test_metrics(self):
        test_set = self.dataset.get('test')['text']
        test_set_labels = np.array(self.dataset.get('test')['label'])
        preds = self.pipeline(test_set, return_all_scores=False)
        predictions = []
        for p in preds:
            predictions.append(int(p['label'].replace('LABEL_', '')))
        false_predicted_indexes = test_set_labels != np.array(predictions)
        acc = load_metric('accuracy')
        acc_val = acc.compute(predictions=predictions, references=test_set_labels)
        prec = load_metric('precision')
        prec_val = prec.compute(predictions=predictions, references=test_set_labels, average='macro')
        rec = load_metric('recall')
        rec_val = rec.compute(predictions=predictions, references=test_set_labels, average='macro')
        f1 = load_metric('f1')
        f1_val = f1.compute(predictions=predictions, references=test_set_labels, average='macro')
        print('Test set results:')
        print(f'Accuracy: {acc_val}')
        print(f'Precision: {prec_val}')
        print(f'Recall: {rec_val}')
        print(f'F1 score: {f1_val}')
        return false_predicted_indexes

    def get_random_samples(self, n=10, split='train', label=None, should_contain_word='') -> (list, list):
        """returns the samples and their labels as builtin python lists: (list, list)
        Parameters
        ----------
        n : int
            the number of samples to be retrieved, The default is 10
        split : str
            can be 'train', 'validation', or 'test'.  which split to get samples from. The default is 'train'
        label: int,
            0 or 1, indicating the label of to be returned samples, defaults to None, if set to 0 or 1 the random
            samples will be sampled from the rows with that label.
        should_contain_word : str
            the word required to be in the samples. If the required word is not in any of the entries, then we return an
            empty list, The default is ''
        """
        ds = self.dataset.get(split)
        if should_contain_word != '':
            ds = ds.filter(lambda row: should_contain_word in row['text'])
        # print(ds)
        if label is not None:
            assert label in [0, 1], 'please use only 0 or 1 as labels'
            ds = ds.filter(lambda row: row['label'] == label)

        if len(ds) > 1:
            indexes = range(0, len(ds) - 1)
            indexes = np.random.choice(indexes, size=n, replace=False)
            print(f'Getting the indexes: {indexes}')
            # can use shuffle and pick the first 200 as well, i.e., samples.shuffle()[:200]
            random_samples = ds.select(indices=indexes)
        else:
            random_samples = ds
        # now we need to transform Huggingface dataset to a python list
        random_samples_pd = random_samples.to_pandas()
        return random_samples_pd['text'].values.tolist(), random_samples_pd['label'].values.tolist()

    def explain_samples(self, samples: list, labels: list, text_plot=True, bar_plot=True, n_most_important_tokens=10,
                        verbose=False, save_as=None):
        """returns the shap values of random samples
        Parameters
        ----------
        samples: list(str)
            strings to be explained
        labels: list(int)
            labels of samples
        text_plot: bool
            whether to show the shap.text_plot() of the samples. The default is True
        bar_plot: bool
            whether to show the most important n_most_important_tokens tokens. The default is True
        n_most_important_tokens: int
            number of tokens to display in the bar plot. The default is 10
        verbose: bool,
            whether to output predictions
        save_as: str,
            if set to none does not save, if set to any str value, saves barplot and forceplot under
            plot_images/<save_as>.pdf
        """
        shap_values = self.explainer(samples)
        print(f'labels: {labels}')
        for i, val in enumerate(shap_values):
            pred = self.predict_sample(samples[i], labels[i], verbose=verbose)
            if bar_plot:
                barplot_first_n_largest_shap_values(val, pred, n=n_most_important_tokens, save_as=save_as)
            if verbose:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if text_plot:
                if save_as is not None:
                    # image_html = shap.plots.text(val[:, pred], display=False)
                    image_html = shap.plots.text(val, display=False)
                    with open(f'plot_images/{save_as}_forceplot.html', 'w') as file:
                        file.write(image_html)
                    img_html = HTML(image_html)
                    ipython_display(img_html)
                else:
                    # shap.plots.text(val[:, pred])
                    shap.plots.text(val)

            if verbose:
                print('#############################################################################################')
        return shap_values

    def predict_sample(self, sample: str, label=None, verbose=False) -> int:
        """convenience method for printing out the prediction probabilities
        Parameters
        ----------
        sample: str
            the sample to be predicted by the model
        label: int
            the actual label of the sample. leave empty when predicting a sample without a label, i.e., test data
        verbose: bool,
            whether to print out the prediction vs actual information, the default is True
        """
        pred = self.pipeline([sample])[0]
        print(pred)
        fake_prob = pred[0]['score']
        real_prob = pred[1]['score']
        if verbose:
            print('###################################################################################################')
            print(f'Predicted fake with {fake_prob}')
            print(f'Predicted real with {real_prob}')
            if label is not None:
                print(f'The actual value is {self.label_mappings[label]}')
            print('---------------------------------------------------------------------------------------------------')
        return 1 if real_prob > fake_prob else 0

    @staticmethod
    def perturb_sample(sample: str, perturbation_type='add', position=0, target_string=None,
                       new_string=None, replace_all=True, replace_until_position=0):
        """
        when perturbation_type is 'add' then the method adds new_string to the given perturbation_location
        when perturbation_type is 'delete' then the method removes target_string from the given sample
        when perturbation_type is 'replace' then the method replaces the target_string with new_string.
        Parameters
        ----------
        sample: str
            string to be perturbed
        perturbation_type: str
            one of 'add', 'delete', 'replace', type of perturbation method
        position: int
            index of which occurrence to remove, if 0 then add to the beginning, if -1 add to the end
            note that this position is the index of the characters not tokenized words.
        target_string: str
            the target string to be replaced
        new_string: str
            the new string that will either
            i. be added to perturbation_location of the sample
            ii. replace the target_string
        replace_all: bool
            if True, and if perturbation_type is 'replace' or 'delete' then replaces/deletes all occurrences.
            if False, replaces/deletes the first occurrence
        replace_until_position: int
            starting from the first, until how many occurrences should the target_string in sample be replaced/deleted.
        """
        perturbation_types = ['add', 'delete', 'replace']
        assert perturbation_type in perturbation_types, f'parameter perturbation_types can only take values: ' \
                                                        f'{perturbation_types}'
        assert position.__abs__() < len(
            sample), f'the given parameter position is an out of range index: {position} <! {len(sample)}'
        if perturbation_type == 'add':
            if position == 0:
                return new_string + sample
            elif position == -1:
                return sample + new_string
            else:
                return sample[0:position] + new_string + sample[position:]
        elif perturbation_type == 'delete':
            matches = re.findall(fr'{target_string}', sample)
            assert len(matches) > 0, 'target_string can not be found in the sample.'
            return sample.replace(target_string, '') if replace_until_position <= 0 or replace_all \
                else sample.replace(target_string, '', replace_until_position)
        else:
            assert target_string is not None and target_string != '', 'target_string should have a value, can not be ' \
                                                                      'None or empty string ("")'
            matches = re.findall(fr'{target_string}', sample)
            assert len(matches) > 0, 'target_string can not be found in the sample.'
            return sample.replace(target_string, new_string) if replace_until_position <= 0 or replace_all \
                else sample.replace(target_string, new_string, replace_until_position)


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    """
    convenience method to force transformers pipeline to truncate the texts to 512 words
    """
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs
