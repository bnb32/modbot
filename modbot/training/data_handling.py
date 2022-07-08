"""Data handling module"""
import numpy as np
import pandas as pd

from keras import utils
import tensorflow as tf

from modbot.utilities.logging import get_logger

logger = get_logger()


class TextGenerator(utils.Sequence):
    """Generator class for text batching"""
    def __init__(self, X, batch_size=64):
        self.batch_size = batch_size
        self.X = X
        self._i = 0
        self.n_batches = np.int(np.ceil(len(X) / batch_size))
        self.chunks = np.array_split(np.arange(len(X)), self.n_batches)
        self.chunks = [slice(c[0], c[-1] + 1) for c in self.chunks]

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.n_batches

    def __getitem__(self, i):
        X = self.X[self.chunks[i]]
        return X

    def __next__(self):
        if self._i < self.n_batches:
            X = self.X[self.chunks[self._i]]
            self._i += 1
            return X
        else:
            raise StopIteration


class DataGenerator(utils.Sequence):
    """Generator class for training over batches"""
    def __init__(self, X, Y, **kwargs):
        """Initialize data generator which provides generators for text and
        targets for training and evaluation

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of texts
        Y : pd.DataFrame
            Pandas dataframe of labels for the corresponding texts
        """
        self.batch_size = kwargs.get('batch_size', 64)
        self.offensive_weight = kwargs.get('offensive_weight', None)
        n_batches = kwargs.get('n_batches', None)
        sample_size = kwargs.get('sample_size', None)

        self.X, self.Y = self.sample(X, Y, self.offensive_weight,
                                     sample_size=sample_size)
        self.indices = pd.DataFrame({'index': np.arange(len(self.Y))})
        self.n_batches = (n_batches if n_batches is not None
                          else np.int(np.ceil(len(self.Y) / self.batch_size)))
        self.chunks = np.array_split(self.indices['index'], self.n_batches)
        self._i = 0

        one_count = list(self.Y).count(1)
        zero_count = list(self.Y).count(0)
        frac = one_count / (zero_count + one_count)

        logger.info(f'Using batch_size={self.batch_size}, '
                    f'n_batches={self.n_batches}, sample_size={sample_size}, '
                    f'offensive_weight={round(frac, 3)}')

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.n_batches

    def __getitem__(self, i):
        X, Y = self.get_batch(i)
        return X, Y

    @classmethod
    def sample(cls, X, Y, offensive_weight=None, sample_size=None):
        """Sample data as per offensive weight

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of texts
        Y : pd.DataFrame
            Pandas dataframe of labels for the corresponding texts
        offensive_weight : float
            Desired ratio of one labels to the total number of labels
        sample_size : int
            Desired sample size. Corresponds to the size of the returned
            texts and labels

        Returns
        -------
        X : pd.DataFrame
            Pandas dataframe of texts with the requested sample size and
            offensive weight
        Y : pd.DataFrame
            Pandas dataframe of labels for the corresponding texts with the
            requested size and offensive weight
        """
        if offensive_weight is None:
            return X, Y

        sample_size = len(Y) if sample_size is None else sample_size
        indices = pd.DataFrame({'index': np.arange(len(Y))})
        zero_weight = 1 - offensive_weight
        n_zeros = int(zero_weight * sample_size)
        n_ones = sample_size - n_zeros

        msg = ('Must have at least one sample for each class. Received a '
               f'sample_size and val_split that resulted in n_ones={n_ones} '
               f'and n_zeros={n_zeros}. Increase the sample_size or the '
               'val_split.')
        assert n_ones > 0 and n_zeros > 0, msg

        zeros = indices['index'][Y == 0].sample(
            n=n_zeros, random_state=42, replace=True)
        ones = indices['index'][Y == 1].sample(
            n=n_ones, random_state=42, replace=True)
        new_indices = pd.concat([ones, zeros])
        new_indices = new_indices.sample(frac=1)
        X = X[new_indices].reset_index(drop=True)
        Y = Y[new_indices].reset_index(drop=True)
        return X, Y

    def get_random_batch(self, _):
        """Get batches of randomly selected texts and targets with specified
        weights

        Returns
        -------
        X : pd.DataFrame
            Pandas dataframe of texts with the number of samples equal to the
            batch size. The texts are randomly selected from the initially
            sampled dataframe of texts.
        Y : pd.DataFrame
            Pandas dataframe of labels with the number of samples equal to the
            batch size. The labels correspond to the randomly selected texts.
        """
        indices = self.indices['index'].sample(n=self.batch_size,
                                               random_state=42, replace=True)
        X = np.array(self.X[indices])
        Y = np.array(self.Y[indices])
        return X, Y

    def get_batch(self, i):
        """Get batch from the i-th chunk of training data

        Parameters
        ----------
        i : int
            Index of chunk used to select slice of full dataframe
        """
        return self.get_deterministic_batch(i)

    def get_deterministic_batch(self, i):
        """Get batches of randomly selected texts and targets

        Parameters
        ----------
        i : int
            Index of chunk used to select slice of full dataframe

        Returns
        -------
        X : pd.DataFrame
            Pandas dataframe of texts with the number of samples equal to the
            batch size. The texts are selected from the i-th chunk of initially
            sampled dataframe of texts.
        Y : pd.DataFrame
            Pandas dataframe of labels with the number of samples equal to the
            batch size. The labels correspond to the i-th chunk of selected
            texts.
        """
        X = np.array(self.X[self.chunks[i]])
        Y = np.array(self.Y[self.chunks[i]])
        return X, Y

    def __next__(self):
        if self._i < self.n_batches:
            X, Y = self.get_batch(self._i)
            self._i += 1
            return X, Y
        else:
            raise StopIteration

    def transform(self, function):
        """Transform texts with provided function

        Parameters
        ----------
        function : model.transform
            Transformation routine from model
        """
        self.X = function(self.X)


@tf.function
def char_split(text):
    """Split function for vectorization layer"""
    return tf.strings.unicode_split(text, input_encoding='UTF-8',
                                    errors='replace')
