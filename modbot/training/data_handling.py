"""Data handling module"""
from attr import has
import numpy as np
import pandas as pd
import dask.dataframe as dd

from keras import utils
import tensorflow as tf

from modbot.utilities.logging import get_logger

logger = get_logger()


class DataGenerator(utils.Sequence):
    """Generator class for batching"""
    def __init__(self, df, **kwargs):
        """Initialize the generator

        Parameters
        ----------
        df : list
            Dataframe to use for batching
        kwargs : dict
            Optional keyword arguments
        """
        self.batch_size = kwargs.get('batch_size', 64)
        self.df = df
        self._chunk_size = kwargs.get('chunk_size', 5)
        self._n_batches = kwargs.get('n_batches', None)
        self._n_samples = None
        self._indices = None
        self._chunks = None
        self._batch_chunks = None
        self._i = 0

    @property
    def n_samples(self):
        """Number of data samples"""
        if self._n_samples is None:
            self._n_samples = len(self.df)
        return self._n_samples

    @property
    def indices(self):
        """Indices for data samples"""
        if self._indices is None:
            if hasattr(self.df, 'index'):
                self._indices = pd.Series(self.df.index)
            else:
                self._indices = pd.Series(list(range(self.n_samples)))
        return self._indices

    @indices.setter
    def indices(self, indices):
        """Update indices attribute"""
        self._indices = indices

    @property
    def chunk_size(self):
        """Get chunk size for splitting data loading"""
        return np.min([self.n_batches, self._chunk_size])

    @property
    def n_chunks(self):
        """Get number of chunks to divide full dataset into for smaller
        reads"""
        return int(np.ceil(self.n_batches / self.chunk_size))

    @property
    def n_batches(self):
        """Get number of batches based on batch size"""
        if self._n_batches is None:
            self._n_batches = int(np.ceil(len(self.df) / self.batch_size))
        return self._n_batches

    @property
    def batch_chunks(self):
        """Get index chunks for batching. Each chunk corresponds to a batch"""
        if self._batch_chunks is None:
            self._batch_chunks = np.array_split(self.indices, self.n_batches)
        return self._batch_chunks

    @property
    def chunks(self):
        """Get dataset chunks. Used to only keep part of full dataset sample in
        memory."""
        if self._chunks is None:
            self._chunks = np.array_split(self.indices, self.n_chunks)
        return self._chunks

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.n_batches

    def __getitem__(self, i):
        return self.get_deterministic_batch(i)

    def __next__(self):
        if self._i < self.n_batches:
            df = self.__getitem__(self._i)
            self._i += 1
            return df
        else:
            raise StopIteration

    def get_random_batch(self, _):
        """Get batches of randomly selected texts and targets with specified
        weights

        Returns
        -------
        arrs : list
            List of randomly selected data batches
        """
        indices = self.indices.sample(n=self.batch_size, random_state=42,
                                      replace=True)
        return self.df.loc[indices]

    def get_deterministic_batch(self, i):
        """Get batches of randomly selected texts and targets

        Parameters
        ----------
        i : int
            Index of chunk used to select slice of full dataframe

        Returns
        -------
        arrs : list
            List of data batches
        """
        return self.df.loc[self.chunks[i]]


class WeightedGenerator(DataGenerator):
    """Generator class for training over batches"""
    def __init__(self, df, **kwargs):
        """Initialize data generator which provides generators for text and
        targets for training and evaluation

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe of texts and labels
        """
        self._one_count = None
        self._zero_count = None
        offensive_weight = kwargs.get('offensive_weight', None)
        sample_size = kwargs.get('sample_size', None)
        indices = self.sample_indices(df['is_offensive'],
                                      offensive_weight=offensive_weight,
                                      sample_size=sample_size)
        df = df.loc[indices]
        if hasattr(df, 'compute'):
            df = df.compute()
        df.index = list(range(len(indices)))
        super().__init__(df, **kwargs)
        frac = self.one_count / (self.zero_count + self.one_count)
        logger.info(f'Using batch_size={self.batch_size}, '
                    f'n_batches={self.n_batches}, sample_size={sample_size}, '
                    f'n_chunks={self.n_chunks}, chunk_size={self.chunk_size}, '
                    f'offensive_weight={round(frac, 3)}')

    @property
    def one_count(self):
        """Get number of samples with target=1"""
        if self._one_count is None:
            self._one_count = len(self.Y[self.Y == 1])
        return self._one_count

    @property
    def zero_count(self):
        """Get number of samples with target=0"""
        if self._zero_count is None:
            self._zero_count = len(self.Y[self.Y == 0])
        return self._zero_count

    @property
    def X(self):
        """alias for texts"""
        if hasattr(self.df, 'compute'):
            return self.df['text'].compute()
        else:
            return self.df['text']

    @property
    def Y(self):
        """alias for targets"""
        if hasattr(self.df, 'compute'):
            return self.df['is_offensive'].compute()
        else:
            return self.df['is_offensive']

    @classmethod
    def sample_indices(cls, Y, offensive_weight=None, sample_size=None):
        """Sample data with weights as per offensive weight

        Parameters
        ----------
        Y : pd.DataFrame
            Pandas dataframe of labels for the corresponding texts
        offensive_weight : float
            Desired ratio of one labels to the total number of labels
        sample_size : int
            Desired sample size. Corresponds to the size of the returned
            texts and labels

        Returns
        -------
        indices : np.array
            Indices of the samples such that X[indices] is the requested sample
            size with the requested offensive_weight
        """
        if hasattr(Y, 'compute'):
            Y = Y.compute()
        one_count = len(Y[Y == 1])
        zero_count = len(Y[Y == 0])
        if offensive_weight is None:
            offensive_weight = one_count / (zero_count + one_count)
        sample_size = len(Y) if sample_size is None else sample_size
        wgt = np.max([offensive_weight, 1 / sample_size])
        wgt_0 = (1 - wgt) / zero_count
        wgt_1 = wgt / one_count
        weights = Y.apply(lambda x: wgt_0 if x == 0 else wgt_1)
        samples = Y.sample(n=sample_size, weights=weights, replace=True,
                           random_state=42)
        samples = samples.sample(frac=1)
        return list(samples.index)

    def transform(self, function):
        """Transform texts"""
        self.df['text'] = function(self.df['text'])

    def __next__(self):
        if self._i < self.n_batches:
            df = self.__getitem__(self._i)
            self._i += 1
            return np.array(df['text']), np.array(df['is_offensive'])
        else:
            raise StopIteration


@tf.function
def char_split(text):
    """Split function for vectorization layer"""
    return tf.strings.unicode_split(text, input_encoding='UTF-8',
                                    errors='replace')
