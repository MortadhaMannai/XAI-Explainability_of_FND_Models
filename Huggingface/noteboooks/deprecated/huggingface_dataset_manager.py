from os import path

import numpy as np
import pandas as pd

from Huggingface.noteboooks.utils import TransformersModelTypeEnum, DATA_DIR, DatasetTypeEnum


class HuggingfaceDatasetManager:
    """
    deprecated class
    handles loading, sampling, preparing of the data for model explanation
    evaluate real as True and troll as fake when working with DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT
    """

    def __init__(self, dataset_type: DatasetTypeEnum):

        self.shuffled_df = None
        true_news_file_dir = path.join(DATA_DIR, dataset_type.value['TRUE_NEWS_DIR'])
        self.true_news_df = self._load_dataframe(true_news_file_dir)

        fake_news_file_dir = path.join(DATA_DIR, dataset_type.value['FAKE_NEWS_DIR'])
        self.fake_news_df = self._load_dataframe(fake_news_file_dir)

        self.text_colname = dataset_type.value['TEXT_COLNAME']
        self.text_col_idx = np.where(self.true_news_df.columns.values == self.text_colname)[0][0]
        if dataset_type == DatasetTypeEnum.KAGGLE_FAKE_NEWS:
            self.title_colname = dataset_type.value['TITLE_COLNAME']
            self.title_col_idx = np.where(self.true_news_df.columns.values == self.title_colname)[0][0]

    @staticmethod
    def _load_dataframe(dir: str):
        print(f"Loading instances from dir: {dir}")
        df = pd.read_csv(dir)
        print(f"Loaded {len(df)} instances from {dir}")
        return df

    def _fetch_rows_with_text(self, text: str, from_true_news: bool):
        """
        method filters the dataset according to the existence of the value of text parameter in the "text" column of the
        dataframe
        """
        df = self.true_news_df if from_true_news else self.fake_news_df
        idxs = df[self.text_colname].map(lambda x: text in x)
        return df[idxs]

    def _fetch_samples(self, dataframe: pd.DataFrame, sample_count: int, sample_random: bool,
                       model_type: TransformersModelTypeEnum):
        # if the indexes are not randomized the first "sample_count" rows will be selected.
        self.idxs = np.random.randint(low=0, high=len(dataframe) - 1,
                                      size=sample_count).tolist() if sample_random else list(range(0, sample_count))
        # print(f"Getting the following indexes: {idxs}")
        return self.fetch_latest_samples(model_type)
        # if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
        #    return dataframe.iloc[self.idxs].apply(self._transform, axis=1)[self.text_colname].values.tolist()
        # return dataframe[self.text_colname].iloc[self.idxs].values.tolist()

    def _transform(self, row):
        title_token = '<title> '
        content_token = ' <content> '
        end_token = ' <end>'
        text_title_part = title_token + row[self.title_col_idx]
        text_content_part = content_token + row[self.text_col_idx]
        row[self.text_col_idx] = text_title_part + text_content_part + end_token
        return row

    def _fetch_samples_with_text(self, from_true_news: bool, text: str, sample_count: int, sample_random: bool):
        df = self._fetch_rows_with_text(text, from_true_news=from_true_news)
        return self._fetch_samples(df, sample_count, sample_random)

    def fetch_true_samples_with_text(self, text: str, sample_count=3, sample_random=True):
        return self._fetch_samples_with_text(True, text, sample_count, sample_random)

    def fetch_fake_samples_with_text(self, text: str, sample_count=3, sample_random=True):
        return self._fetch_samples_with_text(False, text, sample_count, sample_random)

    def fetch_true_samples(self, sample_count=3, sample_random=True):
        return self._fetch_samples(self.true_news_df, sample_count, sample_random)

    def fetch_fake_samples(self, sample_count=3, sample_random=True):
        return self._fetch_samples(self.fake_news_df, sample_count, sample_random)

    def fetch_random_samples(self, model_type: TransformersModelTypeEnum, sample_count=200):
        # add labels for explanation
        true_samples_with_labels = self.true_news_df.copy().sample(int(sample_count / 2))
        true_samples_with_labels['label'] = 1
        fake_samples_with_labels = self.fake_news_df.copy().sample(int(sample_count / 2))
        fake_samples_with_labels['label'] = 0

        # concatenate dataframes then shuffle
        self.shuffled_df = pd.concat([true_samples_with_labels, fake_samples_with_labels]).sample(frac=1)

        return self.shuffled_df['label'], self._fetch_samples(self.shuffled_df, sample_count, False, model_type)

    def fetch_latest_samples(self, model_type: TransformersModelTypeEnum):
        """
        fetch_random_samples should run before this method so that this method can recognize the last sampled indexes.
        """
        assert self.idxs is not None, 'self.idxs is None, you probably did not run fetch_random_samples.'
        assert self.shuffled_df is not None, 'self.shuffled_df is None, you probably did not run fetch_random_samples.'

        if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
            return self.shuffled_df.iloc[self.idxs].apply(self._transform, axis=1)[self.text_colname].values.tolist()
        return self.shuffled_df['label'], self.shuffled_df[self.text_colname].iloc[self.idxs].values.tolist()
