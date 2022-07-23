"""Data handling module"""
from attr import has
import numpy as np
import pandas as pd

from keras import utils
import tensorflow as tf

from modbot.utilities.logging import get_logger

logger = get_logger()


class DataGenerator(utils.Sequence):
    """Generator class for batching"""
    def __init__(self, arrs, **kwargs):
        """Initialize the generator

        Parameters
        ----------
        arrs : list
            List of pd.Dataframe objects to use for batching
        kwargs : dict
            Optional keyword arguments
        """
        self.batch_size = kwargs.get('batch_size', 64)
        self.arrs = arrs
        self._n_batches = kwargs.get('n_batches', None)
        self._n_samples = None
        self._indices = None
        self._chunks = None
        self._i = 0

    @property
    def n_samples(self):
        """Number of data samples"""
        if self._n_samples is None:
            self._n_samples = len(self.arrs[0])
        return self._n_samples

    @property
    def indices(self):
        """Indices for data samples"""
        if self._indices is None:
            if hasattr(self.arrs[0], 'index'):
                self._indices = pd.Series(self.arrs[0].index)
            else:
                self._indices = pd.Series(list(range(self.n_samples)))
        return self._indices

    @indices.setter
    def indices(self, indices):
        """Update indices attribute"""
        self._indices = indices

    @property
    def n_batches(self):
        """Get number of batches based on batch size"""
        if self._n_batches is None:
            self._n_batches = int(np.ceil(len(self.arrs[0]) / self.batch_size))
        return self._n_batches

    @property
    def chunks(self):
        """Get index chunks for batching. Each chunk corresponds to a batch"""
        if self._chunks is None:
            self._chunks = np.array_split(self.indices, self.n_batches)
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
            arrs = self.__getitem__(self._i)
            self._i += 1
            return arrs
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
        arrs = [arr[indices] for arr in self.arrs]
        return *arrs,

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
        arrs = [np.array(arr[self.chunks[i]]) for arr in self.arrs]
        return *arrs,


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
        super().__init__([df['text'][indices], df['is_offensive'][indices]],
                         **kwargs)
        frac = self.one_count / (self.zero_count + self.one_count)
        logger.info(f'Using batch_size={self.batch_size}, '
                    f'n_batches={self.n_batches}, sample_size={sample_size}, '
                    f'offensive_weight={round(frac, 3)}')

    @property
    def one_count(self):
        """Get number of samples with target=1"""
        if self._one_count is None:
            self._one_count = list(self.Y).count(1)
        return self._one_count

    @property
    def zero_count(self):
        """Get number of samples with target=0"""
        if self._zero_count is None:
            self._zero_count = list(self.Y).count(0)
        return self._zero_count

    @property
    def X(self):
        """alias for texts"""
        return self.arrs[0]

    @property
    def Y(self):
        """alias for targets"""
        return self.arrs[1]

    @X.setter
    def X(self, value):
        self.arrs[0] = value

    @Y.setter
    def Y(self, value):
        self.arrs[1] = value

    @classmethod
    def sample_indices(cls, Y, offensive_weight=None, sample_size=None):
        """Sample data as per offensive weight

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
        if offensive_weight is None:
            one_count = list(Y).count(1)
            zero_count = list(Y).count(0)
            offensive_weight = one_count / (zero_count + one_count)
        sample_size = len(Y) if sample_size is None else sample_size
        if hasattr(Y, 'index'):
            indices = pd.Series(Y.index)
        else:
            indices = pd.Series(list(range(len(Y))))
        zero_weight = 1 - offensive_weight
        n_zeros = int(zero_weight * sample_size)
        n_ones = sample_size - n_zeros

        msg = ('Must have at least one sample for each class. Received a '
               f'sample_size and test_split that resulted in n_ones={n_ones} '
               f'and n_zeros={n_zeros}. Increase sample_size or test_split.')
        assert n_ones > 0 and n_zeros > 0, msg

        zeros = indices[Y == 0].sample(n=n_zeros, random_state=42,
                                       replace=True)
        ones = indices[Y == 1].sample(n=n_ones, random_state=42,
                                      replace=True)
        new_indices = pd.concat([ones, zeros])
        new_indices = new_indices.sample(frac=1)
        return new_indices

    def transform(self, function):
        """Transform texts with provided function

        Parameters
        ----------
        function : model.transform
            Transformation routine from model
        """
        self.arrs[0] = function(self.arrs[0])


@tf.function
def char_split(text):
    """Split function for vectorization layer"""
    return tf.strings.unicode_split(text, input_encoding='UTF-8',
                                    errors='replace')
