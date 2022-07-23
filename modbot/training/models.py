"""Models"""
from random import random
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
import os
import tensorflow as tf
import pickle
from inspect import signature
from abc import ABC, abstractmethod
import pprint
from tqdm import tqdm
import copy
import joblib
import re
import json
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text  # pylint: disable=unused-import # noqa: F401

from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import (TensorDataset, DataLoader, SequentialSampler)
from torch.optim import AdamW

from transformers import (BertModel, get_linear_schedule_with_warmup)
from transformers import BertTokenizerFast as BertTokenizer

from keras import Sequential, losses, optimizers, callbacks, layers
from keras.models import load_model
from sklearn.metrics import accuracy_score
import sklearn.feature_extraction.text as ft
from sklearn import svm
import sklearn.calibration as cal
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             jaccard_score, f1_score)
from dask_ml.cluster import KMeans
from dask_ml.model_selection import train_test_split
from scipy import sparse

from modbot.training.vectorizers import TokVectorizer
from modbot.training.data_handling import (DataGenerator, WeightedGenerator)
from modbot.utilities.utilities import curvature
from modbot.utilities.logging import get_logger
from modbot import preprocessing as pp
from modbot import BERT_ENCODER, BERT_PREPROCESS

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

logger = get_logger()


def get_model_class(model_type):
    """Get model class from string and check if valid model type

    Parameters
    ----------
    model_type : str
        A valid model class

    Returns
    -------
    MODEL_CLASS
        A valid model class
    """
    valid_models = {'CNN': CNN, 'LSTM': LSTM, 'SVM': SVM, 'BERT': BERT,
                    'BERT_CNN': BertCNN, 'BERT_LSTM': BertLSTM,
                    'BERT_CNN_TORCH': BertCnnTorch,
                    'BERT_CNN_LSTM': BertCnnLstm,
                    'BERT_LSTM_CNN': BertLstmCnn}
    check = model_type.upper() in valid_models
    msg = (f'Can only load {valid_models.keys()} models. '
           f'Received {model_type}')
    assert check, msg
    return valid_models[model_type.upper()]


class ModerationModel(ABC):
    """Base moderation model class"""

    def __init__(self, texts=None, model=None, **kwargs):
        self.clf = None
        self.vectorizer = None
        self.sample_size = None if texts is None else len(texts)
        self.model = model
        self.train_gen = None
        self.test_gen = None
        self.history = None
        self.X_test = None
        self.Y_test = None
        self.kwargs = kwargs

    @property
    @abstractmethod
    def __name__(self):
        """Model name"""

    @staticmethod
    def clean_text(text):
        """Clean single text so it is utf-8 compliant

        Parameters
        ----------
        text : str
            Text string to clean

        Returns
        -------
        str
            Cleaned text string
        """
        return text.encode('ascii', 'replace').decode()

    @classmethod
    def clean_texts(cls, X):
        """Clean texts so they are utf-8 compliant

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of texts

        Returns
        -------
        X : pd.DataFrame
            Pandas dataframe of cleaned texts
        """
        print(X.shape)
        if isinstance(X, pd.Series):
            X = X.apply(cls.clean_text)
        elif isinstance(X, dd.Series):
            X = X.apply(cls.clean_text, meta=('text', 'object'))
        else:
            for i, x in enumerate(X):
                X[i] = cls.clean_text(x)
        print(X.shape)
        return X

    @classmethod
    def load_data(cls, data_file):
        """Load data from csv file

        Parameters
        ----------
        data_file : str
            Path to csv file storing texts and labels

        Returns
        -------
        df : pd.DataFrame
            Pandas dataframe of texts and labels
        """
        logger.info('Reading in data: %s', data_file)
        data = dd.read_csv(data_file).set_index('index')
        data['text'] = data['text'].apply(cls.clean_text,
                                          meta=('text', 'object'))
        return data

    @classmethod
    def split_data(cls, df, test_split=0.1):
        """Split data into training and test sets

        Parameters
        -------
        df : pd.DataFrame
            Pandas dataframe of texts and labels
        test_split : float
            Fraction of full dataset to use for test data

        Returns
        -------
        df_train : pd.DataFrame
            Pandas dataframe of texts and labels for training
        df_test : pd.DataFrame
            Pandas dataframe of texts and labels for testing
        """
        logger.info('Splitting data')
        out = train_test_split(df, test_size=test_split, random_state=42,
                               shuffle=True)
        return out

    def score(self, X, Y):
        """Score model against targets

        Parameters
        -------
        X : pd.DataFrame
            Pandas dataframe of texts
        Y : pd.DataFrame
            Pandas dataframe of labels for the corresponding texts

        Returns
        -------
        float
            Value of accuracy calulated from the correct predictions vs base
            truth
        """
        Y_pred = self.predict(X, verbose=True)
        return accuracy_score(Y, Y_pred)

    def save(self, outpath):
        """Save model

        Parameters
        ----------
        outpath : str
            Path to save model
        """
        if outpath is not None:
            if '.joblib' in outpath:
                model_dir = os.path.dirname(outpath)
            else:
                model_dir = outpath
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            logger.info(f'Saving {self.__name__} model: {outpath}')
            if hasattr(self.clf, 'save'):
                self.clf.save(outpath)
                history_path = os.path.join(model_dir, 'history.csv')
                logger.info(f'Saving history: {history_path}')
                self.history.to_csv(history_path)
            else:
                with open(outpath, 'wb') as fh:
                    pickle.dump(self.model, fh)

            if self.vectorizer is not None:
                vec_path = os.path.join(model_dir, 'vectorizer.pkl')
                if hasattr(self.vectorizer, 'save'):
                    self.vectorizer.save(vec_path)
                else:
                    with open(vec_path, 'wb') as fh:
                        logger.info(f'Saving vectorizer to: {vec_path}')
                        pickle.dump(self.vectorizer, fh)

            self.save_params(outpath, self.kwargs)
        else:
            logger.info(f'Outpath is None. Not saving {self.__name__} model.')

    @classmethod
    def get_data_generators(cls, data_file, **kwargs):
        """Get data generators with correct sizes

        Parameters
        ----------
        data_file : str
            Path to csv file storing texts and labels
        kwargs : dict
            Dictionary with optional keyword parameters. Can include
            sample_size, batch_size, epochs, n_batches.

        Returns
        -------
        train_gen : WeightedGenerator
            WeightedGenerator instance used for training batches
        test_gen : WeightedGenerator
            WeightedGenerator instance used for evaluation batches
        """
        test_split = kwargs.get('test_split', 0.1)
        df = cls.load_data(data_file)
        df_train, df_test = cls.split_data(df, test_split)
        logger.info('Getting training data generator')
        train_sample_size = kwargs.get('sample_size', None)
        train_sample_size = (len(df_train) if train_sample_size is None
                             else int((1 - test_split) * train_sample_size))
        train_kwargs = copy.deepcopy(kwargs)
        train_kwargs.update({'sample_size': train_sample_size})
        logger.info(f'Using train sample size: {train_sample_size}')
        train_gen = WeightedGenerator(df_train, **train_kwargs)
        logger.info('Getting test data generator')
        test_sample_size = kwargs.get('sample_size', None)
        test_sample_size = (len(df_test) if test_sample_size is None
                            else int(test_sample_size - train_sample_size))
        test_kwargs = copy.deepcopy(kwargs)
        test_kwargs.update({'sample_size': test_sample_size})
        logger.info(f'Using test sample size: {test_sample_size}')
        test_gen = WeightedGenerator(df_test, **test_kwargs)
        return train_gen, test_gen

    def get_class_info(self):
        """Get info about class numbers"""
        logger.info(f'Using n_zeros={list(self.train_gen.Y).count(0)} and '
                    f'n_ones={list(self.train_gen.Y).count(1)} for training.')
        logger.info(f'Using n_zeros={list(self.test_gen.Y).count(0)} and '
                    f'n_ones={list(self.test_gen.Y).count(1)} for testing.')

    @classmethod
    def run(cls, data_file, **kwargs):
        """Run model pipeline. Load data, tokenize texts, and train

        Parameters
        ----------
        data_file : str
            Path to csv file storing texts and labels
        kwargs : dict
            Dictionary with optional keyword parameters. Can include
            sample_size, batch_size, model_path, epochs, n_batches. Needs model
            path to include checkpoint saving callback.

        Returns
        -------
        ModerationModel
            Trained and evaluated keras model or svm
        """
        train_gen, test_gen = cls.get_data_generators(data_file, **kwargs)
        sig = signature(cls)
        params = {k: v for k, v in kwargs.items() if k in sig.parameters}
        if 'texts' in signature(cls).parameters:
            model = cls(texts=np.array(train_gen.X).flatten(), **params)
        else:
            model = cls(**params)
        model.train_gen, model.test_gen = train_gen, test_gen
        model.X_test, model.Y_test = test_gen.X, test_gen.Y
        model.train(train_gen, test_gen, **kwargs)
        return model

    @classmethod
    def continue_training(cls, model_path, data_file, **kwargs):
        """Continue training. Load model, load data, tokenize texts, and train.

        Parameters
        ----------
        model_path : str
            Path to model
        data_file : str
            Path to csv file storing texts and labels
        kwargs : dict
            Dictionary with optional keyword parameters. Can include
            sample_size, batch_size, epochs, n_batches.

        Returns
        -------
        keras.Sequential
            Trained sequential and evaluated model
        """
        model = cls.load(model_path)
        train_gen, test_gen = cls.get_data_generators(data_file, **kwargs)
        model.train_gen, model.test_gen = train_gen, test_gen
        model.X_test, model.Y_test = test_gen.X, test_gen.Y
        just_evaluate = kwargs.get('just_evaluate', False)
        kwargs['model_path'] = model_path
        if not just_evaluate:
            model.train(train_gen, test_gen, **kwargs)
        return model

    def predict(self, X, verbose=False):
        """Predict classification

        Parameters
        ----------
        X : ndarray | list | pd.DataFrame
            Set of texts to classify
        verbose : bool
            Whether to show progress bar for predictions

        Returns
        -------
        list
            List of predicted classifications for input texts
        """
        return np.array([int(r[1] > 0.5)
                         for r in self.predict_proba(X, verbose=verbose)])

    def predict_one(self, X, verbose=False):
        """Predict probability of label=1

        Parameters
        ----------
        X : ndarray | list | pd.DataFrame
            Set of texts to classify
        verbose : bool
            Whether to show progress bar for predictions

        Returns
        -------
        list
            List of predicted probability for label=1 for input texts
        """
        return np.array([r[1] for r in self.predict_proba(X, verbose=verbose)])

    def predict_zero(self, X, verbose=False):
        """Predict probability of label=0

        Parameters
        ----------
        X : ndarray | list | pd.DataFrame
            Set of texts to classify
        verbose : bool
            Whether to show progress bar for predictions

        Returns
        -------
        list
            List of predicted probability for label=0 for input texts
        """
        return np.array([r[0] for r in self.predict_proba(X, verbose=verbose)])

    def model_test(self):
        """Test model on some key phrases

        Parameters
        ----------
        model : ModerationModel
        """

        phrases = ['you are so hot',
                   'its so hot outside',
                   'its hot outside',
                   'hot',
                   'youre dumb',
                   'im dumb',
                   'why are you in just chatting',
                   'im just chatting',
                   'hes so annoying',
                   'this bug is annoying',
                   'go fuck yourself',
                   'this song fucking rocks',
                   'she yawned again',
                   'Hey, wanna play chess with me?I',
                   'wanna go out with me?',
                   'suck my dick',
                   'this game sucks',
                   'you look tired',
                   'im so tired',
                   'ass',
                   'you have a nice ass',
                   'this song is ass'
                   ]

        predictions = [round(p, 3) for p in self.predict_one(phrases)]

        test = pd.DataFrame({'text': phrases, 'probs': predictions})
        return test

    def detailed_score(self, test_gen=None, n_matches=10, out_dir=None):
        """Score model and print confusion matrix and multiple other metrics

        Parameters
        ----------
        test_gen : WeightedGenerator
            generator for test data
        n_matches : int
            Number of positive matches to print
        out_dir : str | None
            Path to save scores

        Returns
        -------
        df_scores : pd.DataFrame
            A dataframe containing all model scores
        """
        if test_gen is not None:
            X, Y = test_gen.X, test_gen.Y
        else:
            X, Y = self.X_test, self.Y_test

        pd.set_option('display.max_columns', None)
        logger.info('Getting detailed info on model performance')
        preds = self.predict_one(X, verbose=True)
        discrete_preds = [int(p > 0.5) for p in preds]
        confusion = confusion_matrix(Y, discrete_preds)
        logger.info(f'Getting info on matches across {len(X)} '
                    'samples')
        discrete_matches = (Y.compute() == discrete_preds)
        one_mask = (Y.compute() == 1) & discrete_matches
        zero_mask = (Y.compute() == 0) & discrete_matches
        one_indices = np.where(one_mask)[0]
        np.random.shuffle(one_indices)
        one_indices = one_indices[:n_matches]
        zero_indices = np.where(zero_mask)[0]
        np.random.shuffle(zero_indices)
        zero_indices = zero_indices[:n_matches]
        X_ones = np.array(X)[one_indices]
        X_zeros = np.array(X)[zero_indices]
        X_match = np.concatenate([X_ones, X_zeros])
        one_probs = [round(preds[i], 3) for i in one_indices]
        zero_probs = [round(preds[i], 3) for i in zero_indices]
        probs = np.concatenate([one_probs, zero_probs])
        one_preds = [int(p > 0.5) for p in one_probs]
        zero_preds = [int(p > 0.5) for p in zero_probs]
        preds = np.concatenate([one_preds, zero_preds])
        truth = [1] * len(X_ones) + [0] * len(X_zeros)
        matches = pd.DataFrame({'text': np.array(X_match),
                                'prob': np.array(probs),
                                'preds': preds,
                                'truth': truth})
        logger.info(f'First {int(2 * n_matches)} matches:\n{matches}')

        test_ones = sum(confusion[1][:])
        test_zeros = sum(confusion[0][:])

        df_scores = pd.DataFrame(
            {'precision': precision_score(Y, discrete_preds),
             'recall': recall_score(Y, discrete_preds),
             'jaccard': jaccard_score(Y, discrete_preds),
             'F1': f1_score(Y, discrete_preds),
             'TP': confusion[1][1] / test_ones,
             'FP': confusion[0][1] / test_zeros,
             'TN': confusion[0][0] / test_zeros,
             'FN': confusion[1][0] / test_ones}, index=[0])
        cols = ['precision', 'recall', 'jaccard', 'F1', 'TP', 'TN']
        df_scores["avg score"] = df_scores[cols].values.mean(axis=1)

        if out_dir is not None:
            out_file = os.path.join(out_dir, f'{self.__name__}_scores.csv')
            logger.info(f'Saving model scores to {out_file}')
            df_scores.to_csv(out_file)

        logger.info(f'Scores:\n{df_scores}')
        logger.info(f'Test phrases:\n{self.model_test()}')
        return df_scores

    @staticmethod
    def save_params(outpath, kwargs):
        """Save params to model path

        Parameters
        ----------
        outpath : str
            Path to model
        kwargs : dict
            Dictionary of kwargs used to build model
        """

        if '.joblib' in outpath:
            model_dir = os.path.dirname(outpath)
        else:
            model_dir = outpath
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        params_path = os.path.join(model_dir, 'params.json')
        logger.info(f'Saving params: {params_path}')
        with open(params_path, 'w') as fh:
            params = dict(kwargs)
            json.dump(params, fh)

    @abstractmethod
    def predict_proba(self, X, verbose=False):
        """Predict probability"""

    @abstractmethod
    def train(self, train_gen, test_gen, **kwargs):
        """Train model"""

    @classmethod
    @abstractmethod
    def load(cls, inpath, **kwargs):
        """Load model from path"""

    @abstractmethod
    def transform(self, X):
        """Transform texts"""


class NNmodel(ModerationModel):
    """Base keras model for moderation"""

    #: Max vocab size for tokenizer
    MAX_NB_WORDS = 50000

    #: Embedding dimension size for embdedding layer
    EMBEDDING_DIM = 100

    #: Max sequence length for tokenizer output
    MAX_SEQUENCE_LENGTH = 250

    def __init__(self, texts=None, clf=None, vec=None, **kwargs):
        """Initialize model

        Parameters
        ----------
        texts : list | ndarray, optional
            Set of texts used to get vocabulary, by default None
        clf : keras.Sequential, optional
            Sequential keras model used when loading from saved dir, by default
            None
        vec : TokVectorizer, optional
            TokVectorizer instance. Used when loading model.
        max_len : int, optional
            Max sequence length for building tokenizer, by default None
        model_path : str
            Path to save model checkpoints

        Raises
        ------
        ValueError
            Error raised if neither texts nor clf is provided
        """

        model_path = kwargs.get('model_path', None)
        self.max_len = kwargs.get('max_len', self.MAX_SEQUENCE_LENGTH)
        self.kwargs = kwargs

        if vec is not None:
            self.vectorizer = vec
        else:
            self.vectorizer = None
        self.callbacks = []
        self.callbacks.append(callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      min_delta=0.0001))
        if model_path is not None:
            self.callbacks.append(callbacks.ModelCheckpoint(
                filepath=model_path + '/epoch_{epoch}', monitor='val_accuracy',
                mode='max', save_best_only=True))
        self.X = None
        self.Y = None
        self.X_test = None
        self.Y_test = None
        self.train_gen = None
        self.test_gen = None
        self.history = None

        if texts is None and clf is not None:
            self.model = self.clf = clf
            logger.info(f'Successfully loaded {self.__name__} model.')
        elif texts is not None:
            logger.info(f'Building {self.__name__} model')
            self.model = self.clf = self.build_layers(texts=texts)
            logger.info(f'Successfully built {self.__name__} model.')
        else:
            raise ValueError(f'{self.__name__} model needs either texts or '
                             'clf to be initialized')

        self.layer_names = [layer.name for layer in self.clf.layers]
        logger.info(f'Model layers: {self.layer_names}')

    @property
    def __name__(self):
        """Model name"""
        return "BASE_NN_MODEL"

    @staticmethod
    def standardize_grams(grams):
        """Standarize vocab. Lower and remove punctuation

        Parameters
        ----------
        grams : dict
            Dictionary of grams with values as gram count

        Returns
        -------
        clean_grams : dict
            Dictionary of standardized grams with values as gram count
        """
        clean_grams = {}
        logger.info(f'Standardizing vocabulary with {len(grams)} words')
        for gram in tqdm(grams):
            clean_gram = re.sub(r"[^a-zA-Z0-9]", "", gram).lower()
            clean_grams[clean_gram] = clean_grams.get(clean_gram, 0) + 1
        clean_grams.pop('', None)
        clean_grams.pop('unk', None)
        return clean_grams

    def get_vocab_difference(self, texts):
        """Get difference in direct vocab and adapted vocab

        Parameters
        ----------
        texts : list | ndarray | pd.DataFrame
            Set of texts used to build vocabulary
        """
        direct_vocab = self.standardize_grams(self.get_vocab_direct(texts))
        encoder = layers.TextVectorization(name='encoder')
        encoder = self.construct_vocab(encoder, texts)
        adapted_vocab = encoder.get_vocabulary()
        vocab_diff = set(direct_vocab) ^ set(adapted_vocab)
        logger.info('Difference in direct vocab and adapted vocab: '
                    f'{vocab_diff}')
        logger.info(f'{len(direct_vocab)} words in direct_vocab and '
                    f'{len(adapted_vocab)} words in adapted_vocab')
        logger.info(f'{len(vocab_diff)} words in vocab_diff')

    @staticmethod
    def construct_vocab(encoder, texts):
        """Construct vocab with encoder

        Parameters
        ----------
        encoder : layers.TextVectorization
            TextVectorization layer used to build network
        texts : list | ndarray | pd.DataFrame
            Set of texts used to build vocabulary

        Returns
        -------
        encoder : layers.TextVectorization
            TextVectorization layer used to build network which has been
            adapted to get vocab
        """
        logger.info('Constructing vocabulary')
        text_dataset = DataGenerator([texts])
        encoder.adapt(text_dataset)
        return encoder

    @staticmethod
    def get_most_common_ngrams(vocab, n_min=None, n_max=None):
        """Get most common of each ngram

        Parameters
        ----------
        vocab : list
            List of ngrams which have been built previously
        n_min : int | None
            Minimum size of ngram used when building vocab
        m_max : int | None
            Maximum size of ngram used when building vocab
        """
        if n_min is not None and n_max is not None:
            max_ngrams = {}
            ngram_record = [False] * (n_max + 1 - n_min)
            logger.info('Finding the most frequent of each ngram')
            for i in tqdm(range(n_min, n_max + 1)):
                for word in vocab:
                    if len(word) == i and not ngram_record[i]:
                        max_ngrams[word] = vocab[word]
                        ngram_record[i] = True
            logger.info('The most frequent of each ngram is:\n'
                        f'{pprint.pformat(max_ngrams, indent=1)}')

    @staticmethod
    def get_ngrams(text, n_min, n_max):
        """Create char n grams from text

        Parameters
        ----------
        text : str
            Text string for which to compute ngrams
        n_min : int | None
            Minimum size of ngram used when building vocab
        m_max : int | None
            Maximum size of ngram used when building vocab

        Returns
        -------
        grams : dict
            Dictionary of grams with values as gram count
        """
        grams = {}
        for n in range(n_min, n_max + 1):
            for i in range(len(text) - n):
                gram = text[i: i + n]
                grams[gram] = grams.get(gram, 0) + 1
        return grams

    @staticmethod
    def chunk_words(text):
        """Create char ngrams from word

        Parameters
        ----------
        text : str
            Text string for which to compute ngrams

        Returns
        -------
        grams : dict
            Dictionary of grams with values as gram count
        """
        grams = {}
        for word in text.split():
            for n in range(len(word), 2, -1):
                for i in range(len(word) - n):
                    gram = word[i: i + n]
                    grams[gram] = grams.get(gram, 0) + 1
        return grams

    @staticmethod
    def split_words(text):
        """Split text into words

        Parameters
        ----------
        text : str
            Text string for which to compute ngrams

        Returns
        -------
        grams : dict
            Dictionary of grams with values as gram count
        """
        grams = {}
        for gram in text.split():
            grams[gram] = grams.get(gram, 0) + 1
        return grams

    @classmethod
    def get_vocab_direct(cls, texts, n_min=None, n_max=None,
                         chunk_words=False):
        """Get vocab direct from texts

        texts : list | ndarray | pd.DataFrame
            List of texts for which to compute vocab
        n_min : int | None
            Minimum size of ngram used when building vocab
        m_max : int | None
            Maximum size of ngram used when building vocab
        chunk_words : bool
            Whether to compute ngrams on individual words

        Returns
        -------
        list
            List of words or ngrams
        """
        logger.info('Getting vocab direct from training texts')
        vocab = {}
        for t in tqdm(texts):
            if n_min is not None and n_max is not None:
                grams = cls.get_ngrams(t, n_min, n_max)
            elif chunk_words:
                grams = cls.chunk_words(t)
            else:
                grams = cls.split_words(t)
            for gram in grams:
                vocab[gram] = vocab.get(gram, grams.get(gram, 0)) + 1

        vocab = cls.standardize_grams(vocab)
        vocab = dict(sorted(vocab.items(), reverse=True,
                            key=lambda item: item[1]))
        logger.info(f'Vocabulary has {len(vocab.keys())} words')
        most_frequent = {k: v for i, (k, v)
                         in enumerate(vocab.items()) if i < 10}

        cls.get_most_common_ngrams(vocab, n_min, n_max)

        logger.info('The most frequent 10 words are:\n'
                    f'{pprint.pformat(most_frequent, indent=1)}')
        return list(vocab.keys())

    @abstractmethod
    def build_layers(self, **kwargs):
        """Build model layers"""

    def train(self, train_gen, test_gen, **kwargs):
        """Train model and evaluate

        Parameters
        ----------
        train_gen : WeightedGenerator
            WeightedGenerator instance used for training batches
        test_gen : WeightedGenerator
            WeightedGenerator instance used for evaluation batches
        kwargs : dict
            Dictionary with optional keyword parameters. Can include
            sample_size, batch_size, epochs, n_batches.
        """
        self.get_class_info()
        train_gen.transform(self.transform)
        test_gen.transform(self.transform)
        history = self.fit(train_gen, test_gen, epochs=kwargs.get('epochs', 5))
        self.append_history(history)

    def append_history(self, history):
        """Append history if continuing training or define history attribute

        Parameters
        ----------
        history : dict
            Dictionary containing training history
        """
        if self.history is not None:
            self.history = pd.concat([self.history,
                                      pd.DataFrame(history.history)])
        else:
            self.history = pd.DataFrame(history.history)

    def transform(self, X):
        """Transform texts

        Parameters
        ----------
        X : list | ndarray | pd.DataFrame
            Set of texts to transform before sending to model

        Returns
        -------
        X : list | ndarray | pd.DataFrame
            Set of transformed texts ready to send to model
        """
        return self.clean_texts(X)

    def fit(self, train_gen, test_gen, epochs=5):
        """Fit model

        Parameters
        ----------
        train_gen : WeightedGenerator
            WeightedGenerator instance used for training batches
        test_gen : WeightedGenerator
            WeightedGenerator instance used for evaluation batches
        epochs : int
            Number of epochs to train model

        Returns
        -------
        dict
            Dictionary of training history
        """
        logger.info('Fitting model')
        history = self.clf.fit(train_gen, epochs=epochs,
                               validation_data=test_gen,
                               callbacks=self.callbacks)
        return history

    def predict_proba(self, X, verbose=False):
        """Predict probability

        Parameters
        ----------
        X : ndarray | list | pd.DataFrame
            Set of texts to classify
        verbose : bool
            Whether to show progress bar for predictions

        Returns
        -------
        list
            List of probabilities of having label=1 for input texts
        """
        X_in = np.array(self.transform(X))
        if verbose:
            logger.info(f'Making predictions on {len(X_in)} inputs')
            results = self.clf.predict(X_in, steps=len(X_in))
        else:
            results = self.clf.predict(X_in, steps=len(X_in), verbose=0)
        return [[1 - x[0], x[0]] for x in results]

    def evaluate(self, test_gen):
        """Evaluate model

        Parameters
        ----------
        test_gen : WeightedGenerator
            WeightedGenerator instance used for model evaluation

        Returns
        -------
        list
            List of evaluation results
        """
        logger.info('Evaluating model')
        accr = self.clf.evaluate(test_gen)
        return accr

    @classmethod
    def load(cls, inpath, **kwargs):
        """Load model

        Parameters
        ----------
        inpath : str
            Path from which to load model

        Returns
        -------
            Initialized NNmodel model
        """
        logger.info(f'Loading {cls.__name__} model from {inpath}')
        if '.joblib' in inpath:
            model_dir = os.path.dirname(inpath)
        else:
            model_dir = inpath
        clf = load_model(inpath)
        vec_path = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(vec_path):
            vec = TokVectorizer.load(vec_path)
        else:
            vec = None
        model = cls(clf=clf, vec=vec, model_path=inpath, **kwargs)
        history_path = os.path.join(model_dir, 'history.csv')
        if os.path.exists(history_path):
            model.history = pd.read_csv(history_path)
        return model

    @staticmethod
    def print_eval(accr):
        """Log evaluation

        Parameters
        ----------
        accr : list
            List of evaluation results
        """
        logger.info('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
            accr[0], accr[1]))


class LSTM(NNmodel):
    """LSTM model for moderation"""

    def get_encoder(self, texts, n_min=None, n_max=None, chunk_words=False):
        """Get encoding layer for network input

        Parameters
        ----------
        texts : list | ndarray | pd.DataFrame
            Set of texts used to build vocabulary
        n_min : int | None
            Minimum size of ngram used when building vocab
        m_max : int | None
            Maximum size of ngram used when building vocab
        chunk_words : bool
            Whether to compute ngrams on individual words

        Returns
        -------
        layers.TextVectorization
            TextVectorization layer used to build network
        """
        vocab = self.get_vocab_direct(texts, n_min, n_max, chunk_words)
        encoder = layers.TextVectorization(name='encoder', vocabulary=vocab)
        return encoder

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        texts : list | ndarray | pd.DataFrame
            List of texts from which to build vocab
        n_min : int | None
            Minimum size of ngram used when building vocab
        m_max : int | None
            Maximum size of ngram used when building vocab
        chunk_words : bool
            Whether to compute ngrams on each word

        Returns
        -------
        keras.Sequential
            Sequential model
        """

        texts = kwargs.get('texts', None)
        n_min = kwargs.get('n_min', None)
        n_max = kwargs.get('n_max', None)
        chunk_words = kwargs.get('chunk_words', False)

        msg = (f'Must provide texts to build layers for {self.__name__} model')
        assert texts is not None, msg

        encoder = self.get_encoder(texts, n_min=n_min, n_max=n_max,
                                   chunk_words=chunk_words)
        vocab = encoder.get_vocabulary()
        max_words = len(vocab)
        embedder = layers.Embedding(input_dim=max_words,
                                    output_dim=self.EMBEDDING_DIM)
        model = Sequential()
        model.add(encoder)
        model.add(embedder)
        model.add(layers.SpatialDropout1D(0.2))
        model.add(layers.LSTM(100, dropout=0.2))
        model.add(layers.Dense(1, name='out_layer',
                               activation='sigmoid'))
        model.compile(loss=losses.BinaryCrossentropy(),
                      optimizer=optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        return model

    @property
    def __name__(self):
        """Model name"""
        return 'LSTM'


class CNN(NNmodel):
    """Moderation model using convolution layer"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Sequential
            Keras model
        """
        texts = kwargs.get('texts', None)
        msg = (f'Must provide texts to build layers for {self.__name__} model')
        assert texts is not None, msg

        embedder = layers.Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM,
                                    input_length=self.max_len,
                                    name='token_embedder')
        logger.info(
            f'Initializing CNN model with max sequence len: {self.max_len}')
        self.vectorizer = TokVectorizer(max_len=self.max_len)
        self.vectorizer.fit(texts)
        model = Sequential()
        model.add(layers.Input(name='inputs', shape=[self.max_len]))
        model.add(embedder)
        model.add(layers.Conv1D(self.EMBEDDING_DIM, 3, padding='same',
                                activation='relu'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Flatten())
        model.add(layers.Dense(250, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss=losses.BinaryCrossentropy(),
                      optimizer=optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        return model

    def transform(self, X):
        return self.vectorizer.transform(X)

    @property
    def __name__(self):
        """Model name"""
        return 'CNN'


class BERT(NNmodel):
    """Bert model"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Model
            Keras model
        """

        bert_preprocess, bert_encoder = self.init_bert(**kwargs)
        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        out = layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        out = layers.Dense(1, activation='sigmoid', name="output")(out)
        model = tf.keras.Model(inputs=[text_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def init_bert(self, **kwargs):
        """Initialize BERT

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        bert_preprocess : hub.KerasLayer
            BERT tokenizer to be used in network
        bert_encoder : hub.KerasLayer
            BERT encoder to be used in network
        """

        bert_preproc_path = kwargs.get('bert_preprocess', BERT_PREPROCESS)
        bert_encoder_path = kwargs.get('bert_encoder', BERT_ENCODER)
        logger.info('Getting BERT preprocess layer')
        bert_preprocess = tf_hub.KerasLayer(bert_preproc_path)
        logger.info('Getting BERT encoder layer')
        bert_encoder = tf_hub.KerasLayer(bert_encoder_path, trainable=False,
                                         name='bert_encoder')

        return bert_preprocess, bert_encoder

    @property
    def __name__(self):
        """Model name"""
        return 'BERT'


class BertCNN(BERT):
    """Moderation model with BERT and CNN"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Model
            Keras model
        """

        CONV = layers.Conv2D
        POOL = layers.GlobalMaxPooling2D

        bert_preprocess, bert_encoder = self.init_bert()
        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        out = tf.stack(outputs['encoder_outputs'][-4:], axis=-2)
        conv_args = dict(padding='same', activation='relu')
        out_1 = POOL()(CONV(64, 1, **conv_args)(out))
        out_2 = POOL()(CONV(64, 2, **conv_args)(out))
        out_3 = POOL()(CONV(64, 3, **conv_args)(out))
        out_4 = POOL()(CONV(64, 4, **conv_args)(out))
        out_5 = POOL()(CONV(64, 5, **conv_args)(out))
        out = tf.concat([out_1, out_2, out_3, out_4, out_5], axis=-1)
        out = layers.Flatten()(out)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.Dense(1, activation='sigmoid', name="output")(out)
        model = tf.keras.Model(inputs=[text_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    @property
    def __name__(self):
        """Model name"""
        return "BERT_CNN"


class BertLSTM(BERT):
    """Bert model with LSTM"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Model
            Keras model
        """

        POOL = layers.GlobalMaxPooling1D
        LSTM = layers.LSTM(64, dropout=0.05)

        bert_preprocess, bert_encoder = self.init_bert()
        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        out = tf.stack([LSTM(out) for out in outputs['encoder_outputs'][-4:]],
                       axis=-2)
        out_1 = POOL()(out)
        out_2 = POOL()(out)
        out_3 = POOL()(out)
        out_4 = POOL()(out)
        out_5 = POOL()(out)
        out = tf.concat([out_1, out_2, out_3, out_4, out_5], axis=-1)
        out = layers.Flatten()(out)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.Dense(1, activation='sigmoid', name="output")(out)
        model = tf.keras.Model(inputs=[text_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    @property
    def __name__(self):
        """Model name"""
        return 'BERT_LSTM'


class BertLstmCnn(BERT):
    """Model with Bert, LSTM, and CNN in that order"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Model
            Keras model
        """

        CONV = layers.Conv2D
        POOL = layers.GlobalMaxPooling2D
        LSTM = layers.LSTM(32, dropout=0.05)

        bert_preprocess, bert_encoder = self.init_bert()
        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        out = tf.stack([LSTM(out) for out in outputs['encoder_outputs'][-4:]],
                       axis=-2)
        out = tf.expand_dims(out, axis=0)
        conv_args = dict(padding='same', activation='relu')
        out_1 = POOL()(CONV(64, 1, **conv_args)(out))
        out_2 = POOL()(CONV(64, 2, **conv_args)(out))
        out_3 = POOL()(CONV(64, 3, **conv_args)(out))
        out_4 = POOL()(CONV(64, 4, **conv_args)(out))
        out_5 = POOL()(CONV(64, 5, **conv_args)(out))
        out = tf.concat([out_1, out_2, out_3, out_4, out_5], axis=-1)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.Flatten()(out)
        out = layers.Dense(1, activation='sigmoid', name="output")(out)
        model = tf.keras.Model(inputs=[text_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    @property
    def __name__(self):
        """Model name"""
        return 'BERT_LSTM_CNN'


class BertCnnLstm(BERT):
    """Model with Bert, CNN, and LSTM in that order"""

    def build_layers(self, **kwargs):
        """Build model layers

        Parameters
        ----------
        kwargs : dict
            Dictionary of config parameters

        Returns
        -------
        keras.Model
            Keras model
        """

        CONV = layers.Conv2D
        LSTM = layers.LSTM(32, dropout=0.05)

        bert_preprocess, bert_encoder = self.init_bert()
        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        out = tf.stack(outputs['encoder_outputs'][-4:], axis=-2)
        conv_args = dict(padding='same', activation='relu')
        out_1 = LSTM(tf.reduce_mean(CONV(64, 1, **conv_args)(out), axis=-2))
        out_2 = LSTM(tf.reduce_mean(CONV(64, 2, **conv_args)(out), axis=-2))
        out_3 = LSTM(tf.reduce_mean(CONV(64, 3, **conv_args)(out), axis=-2))
        out_4 = LSTM(tf.reduce_mean(CONV(64, 4, **conv_args)(out), axis=-2))
        out_5 = LSTM(tf.reduce_mean(CONV(64, 5, **conv_args)(out), axis=-2))
        out = tf.concat([out_1, out_2, out_3, out_4, out_5], axis=-1)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.Flatten()(out)
        out = layers.Dense(1, activation='sigmoid', name="output")(out)
        model = tf.keras.Model(inputs=[text_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def combo_layer(outputs, k_size):
        """Combination layer

        Parameters
        ----------
        outputs : list
            List of tensors from BERT
        k_size : int
            Size of convolution kernel

        Returns
        -------
        out : tf.Tensor
            Output of layer
        """
        CONV = layers.Conv1D
        LSTM = layers.LSTM(32, dropout=0.05)
        POOL = layers.GlobalMaxPooling1D

        conv_args = dict(padding='same', activation='relu')
        out = [LSTM(CONV(64, k_size, **conv_args)(out)) for out in outputs]
        return POOL()(tf.stack(out, axis=-2))

    @property
    def __name__(self):
        """Model name"""
        return 'BERT_CNN_LSTM'


class SVM(ModerationModel):
    """Linear SVM model"""

    #: Default parameters for tf-idf, svm, and calibrated classifier
    PARAMS = dict({'stop_words': None, 'tokenizer': None, 'cv': 5,
                   'method': 'sigmoid', 'min_df': 1, 'max_df': 1.0,
                   'analyzer': 'char_wb', 'ngram_range': (1, 8),
                   'smooth_idf': 1, 'sublinear_tf': 1, 'max_iter': 10000,
                   'C': 1})

    def __init__(self, texts=None, model=None, **kwargs):
        """Initialize SVM model

        Parameters
        ----------
        model : sklearn.Pipeline | None
            Previously trained sklearn pipeline object. Need to load saved
            model
        """
        self.kwargs = kwargs
        params = self.PARAMS
        if model is not None:
            self.model = model
            self.vectorizer = model['vectorizer']
            self.clf = model['classifier']
            logger.info(f'Succesfully loaded {self.__name__} model')
        else:
            sig = signature(ft.TfidfVectorizer)
            vparams = {k: v for k, v in params.items() if k in sig.parameters}
            sig = signature(svm.LinearSVC)
            cparams = {k: v for k, v in params.items() if k in sig.parameters}
            sig = signature(cal.CalibratedClassifierCV)
            cvparams = {k: v for k, v in params.items() if k in sig.parameters}
            self.vparams = vparams
            self.cparams = cparams
            self.vectorizer = ft.TfidfVectorizer(**vparams)
            self.clf = svm.LinearSVC(**cparams)
            self.clf = cal.CalibratedClassifierCV(base_estimator=self.clf,
                                                  **cvparams)
            self.model = Pipeline([('vectorizer', self.vectorizer),
                                   ('classifier', self.clf)])
            logger.info('Succesfully built model')
        self.train_gen = None
        self.test_gen = None
        self.X_test = None
        self.Y_test = None

    @property
    def __name__(self):
        """Model name"""
        return 'SVM'

    def train(self, train_gen, test_gen=None, **kwargs):
        """Train model

        Parameters
        ----------
        train_gen : WeightedGenerator
            WeightedGenerator instance used for training batches
        test_gen : WeightedGenerator
            Has no effect. For compliance with LSTM train method
        kwargs : dict
            Has no effect. For compliance with LSTM train method
        """
        self.get_class_info()
        logger.info('Training LinearSVM classifier')
        train_gen.X = train_gen.X.apply(pp.correct_msg)
        self.model.fit(train_gen.X, train_gen.Y)

    def predict_proba(self, X, verbose=False):
        """Predict classification

        Parameters
        ----------
        X : ndarray | list | pd.DataFrame
            Set of texts to classify
        verbose : bool
            Has no effect. For compliance with LSTM method

        Returns
        -------
        list
            List of predicted classifications for input texts
        """
        return self.model.predict_proba(X)

    def transform(self, X):
        """Transform texts

        Parameters
        ----------
        X : list | ndarray | pd.DataFrame
            Set of texts to transform before sending to model

        Returns
        -------
        X : list | ndarray | pd.DataFrame
            Set of transformed texts ready to send to model
        """
        return self.vectorizer.transform(X)

    @classmethod
    def load(cls, inpath, **kwargs):
        """Load SVM model from path

        Parameters
        ----------
        inpath : str
            Path to load model from

        Returns
        -------
        SVM
            Previously trained and saved model
        """
        logger.info(f'Loading {cls.__name__} model from {inpath}')
        model = joblib.load(inpath)
        return cls(model=model, **kwargs)


class KMeansComponent:
    """KMeans model class. With standard model methods."""
    def __init__(self):
        self.pca = TruncatedSVD(n_components=30)
        self.kmeans = KMeans(n_clusters=2, random_state=0)
        self.comps = None
        self.classes = None

    @staticmethod
    def get_optimal_k(df):
        """Calculate the optimal number of clusters from distortion progress"""
        distortions = []
        num_clusters = range(1, 10)
        for k in num_clusters:
            clusters = KMeans(n_clusters=k, random_state=0).fit(df)
            distortions.append(clusters.inertia_)
        return num_clusters[np.argmax(curvature(distortions))]

    @classmethod
    def optimize(cls, comps):
        """Get optimal number of clusters"""
        return cls.get_optimal_k(comps)

    def fit(self, data):
        """Fit model to data"""
        self.comps = self.pca.fit_transform(data)
        self.kmeans = KMeans(n_clusters=self.optimize(self.comps),
                             random_state=0)
        self.kmeans.fit(self.comps)
        return sparse.hstack((data, np.array(self.kmeans.labels_)[:, None]))

    def transform(self, data):
        """Transform data before training"""
        self.comps = self.pca.transform(data)
        self.classes = self.kmeans.predict(self.comps)
        return sparse.hstack((data, np.array(self.classes)[:, None]))

    def fit_transform(self, data):
        """Transform data and fit model"""
        self.comps = self.pca.fit_transform(data)
        self.kmeans = KMeans(n_clusters=self.optimize(self.comps),
                             random_state=0)
        self.kmeans.fit(self.comps)
        return sparse.hstack((data, np.array(self.kmeans.labels_)[:, None]))


class BertCnnTorchModel(nn.Module):
    """Bert Cnn model pytorch implementation"""

    def __init__(self, embed_size):
        super().__init__()
        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = 32
        self.convs1 = nn.ModuleList([nn.Conv2d(4, num_filters, (K, embed_size))
                                     for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.sigmoid = nn.Sigmoid()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',
                                                    output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, x, input_masks, token_type_ids):
        """Forward pass for model"""

        x = self.bert_model(x, attention_mask=input_masks,
                            token_type_ids=token_type_ids)[2][-4:]
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return self.sigmoid(logit)


class TorchModel(NNmodel):
    """Base torch model"""

    SEED = 42
    MAX_SEQUENCE_LENGTH = 64
    DLOADER_ARGS = {'num_workers': 1, 'pin_memory': True}
    LEARNING_RATE = 2e-5
    EMBED_SIZE = 768
    LOSS_FUNCTION = nn.BCELoss()

    def __init__(self, checkpoint=None, embed_size=None, lr=None,
                 device='gpu'):
        embed_size = embed_size if embed_size is not None else self.EMBED_SIZE
        lr = lr if lr is not None else self.LEARNING_RATE
        if torch.cuda.is_available() and device == 'gpu':
            logger.info('**Using GPU for training/inference**')
            self.device = torch.device("cuda:0")
        else:
            logger.info('**Using CPU for training/inference**')
            self.device = torch.device("cpu")

        self.clf = self.build_layers(embed_size)
        self.optimizer = AdamW(self.clf.parameters(), lr=lr, weight_decay=0.9)
        self.best_score = -np.inf
        self.epoch = 0
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def loss_fn(self, preds, truth):
        """Loss function"""
        return self.LOSS_FUNCTION(preds, truth)

    def load_checkpoint(self, checkpoint):
        """Load model from checkpoint"""
        self.clf.load_state_dict(checkpoint['model_state'])
        self.clf.to(self.device)
        self.best_score = checkpoint['best_score']
        self.epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f'Loaded checkpoint with epoch ({self.epoch}) '
                    f'best_score ({round(self.best_score, 3)})')

    @property
    def __name__(self):
        return "BASE_TORCH_MODEL"

    @abstractmethod
    def build_layers(self, embed_size):
        """Build model layers"""

    @abstractmethod
    def transform(self, X, Y=None, max_length=None, batch_size=None):
        """returns dataloader for input to training or prediction methods"""

    def predict_proba(self, X, verbose=False, batch_size=128):
        """Make prediction on input texts"""
        test_dataloader = self.transform(X, batch_size=batch_size)
        self.clf.eval()
        with torch.no_grad():
            preds = []
            if verbose:
                iterator = tqdm(test_dataloader)
            else:
                iterator = test_dataloader
            for batch in iterator:
                out = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_token_type_ids = out
                y_pred = self.clf(b_input_ids, b_input_mask,
                                  b_token_type_ids)
                preds += list(y_pred.cpu().numpy().flatten())

        return [[1 - x, x] for x in preds]

    @staticmethod
    def eval_score(df_scores):
        """Evaluation score to determine whether to save model"""
        tp = float(df_scores['TP'])
        tn = float(df_scores['TN'])
        f1 = float(df_scores['F1'])
        return round(10 * tn + tp + f1, 5)

    def evaluate(self, dev_dataloader, epoch, loss_fn):
        """Evaluate model on test data"""
        val_preds = []
        with torch.no_grad():
            val_loss = 0
            logger.info(f'Evaluating on {len(dev_dataloader)} batches for '
                        f'epoch {epoch}')
            for batch in tqdm(dev_dataloader):
                out = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = out
                y_pred = self.clf(b_input_ids, b_input_mask,
                                  b_token_type_ids)
                loss = loss_fn(y_pred, b_labels.unsqueeze(1))
                y_pred = y_pred.cpu().numpy().flatten()
                val_preds += [int(p >= 0.5) for p in y_pred]
                val_loss += loss.item()
                for param in self.clf.parameters():
                    param.grad = None
        return val_loss, val_preds

    def train(self, train_gen, test_gen, **kwargs):
        """Train pytorch model"""
        epochs = kwargs.get('epochs', 10)
        model_path = kwargs.get('model_path', None)
        batch_size = kwargs.get('batch_size', 128)
        max_length = kwargs.get('max_length', self.MAX_SEQUENCE_LENGTH)
        eval_steps = kwargs.get('eval_steps', 100)
        self.get_class_info()

        logger.info('Encoding training data')
        train_dataloader = self.transform(X=train_gen.X.values.compute(),
                                          Y=train_gen.Y.values.compute(),
                                          max_length=max_length,
                                          batch_size=batch_size)
        self.clf.to(self.device)
        train_losses = []
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.SEED)

        train_batches = len(train_dataloader)
        total_steps = train_batches * epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for param in self.clf.parameters():
            param.grad = None

        logger.info('Starting training')
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch
            train_loss = 0
            self.clf.train(True)
            logger.info(f'Training on {len(train_gen)} batches for '
                        f'epoch {epoch}')
            step_count = 0
            for batch in tqdm(train_dataloader):
                out = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = out
                y_pred = self.clf(b_input_ids, b_input_mask,
                                  b_token_type_ids)
                loss = self.loss_fn(y_pred, b_labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                scheduler.step()
                for param in self.clf.parameters():
                    param.grad = None

                step_count += 1
                eval_check = (step_count % eval_steps == 0
                              or step_count % train_batches == 0)
                if eval_check:
                    self.clf.eval()
                    score = self.eval_score(self.detailed_score(test_gen))
                    msg = f'Epoch: {epoch}. New score: {score}. '
                    msg += f'Old score: {self.best_score}.'
                    logger.info(msg)
                    if score > self.best_score:
                        self.best_score = score
                        self.save(model_path)
                    self.clf.train(True)
            train_losses.append(train_loss)

        if model_path is not None:
            checkpoint = torch.load(model_path)
            self.clf.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.clf.to(self.device)
        self.clf.eval()
        return self.clf

    def save(self, outpath):
        """Save model"""
        if outpath is not None:
            checkpoint = {'model_state': self.clf.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'best_score': self.best_score,
                          'epoch': self.epoch}
            torch.save(checkpoint, outpath)
            logger.info(f'{self.__name__} model saved to {outpath}')
        else:
            logger.info(f'Outpath is None. Not saving {self.__name__} model')

    def transform(self, X, Y=None, max_length=None, batch_size=None):
        """returns dataloader for input to training or prediction methods"""
        max_length = (max_length if max_length is not None
                      else self.MAX_SEQUENCE_LENGTH)
        t = self.tokenizer.batch_encode_plus(list(self.clean_texts(X)),
                                             padding='max_length',
                                             add_special_tokens=True,
                                             max_length=max_length,
                                             return_tensors='pt',
                                             truncation=True)
        if Y is not None:
            out = TensorDataset(t["input_ids"], t["attention_mask"],
                                t["token_type_ids"], torch.FloatTensor(Y))
        else:
            out = TensorDataset(t["input_ids"], t["attention_mask"],
                                t["token_type_ids"])
        sampler = SequentialSampler(out)
        out = DataLoader(out, sampler=sampler, batch_size=batch_size,
                         **self.DLOADER_ARGS)
        return out

    @classmethod
    def load(cls, inpath, **kwargs):
        """Load pytorch model"""
        if inpath is not None:
            logger.info(f'Loading {cls.__name__} model from {inpath}')
            model = cls(checkpoint=torch.load(inpath), **kwargs)
            return model
        else:
            logger.info(f'Inpath is not None. Not loading {cls.__name__} '
                        f'model.')


class BertCnnTorch(TorchModel):
    """Bert Cnn model pytorch implementation"""

    def __init__(self, *args, **kwargs):
        """Initialize model

        Parameters
        ----------
        args
            same args as base class
        kwargs
            same kwargs as base class
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def loss_fn(self, preds, truth):
        fp = torch.sum(preds * (1 - truth)) / len(preds)
        fn = torch.sum((1 - preds) * truth) / len(preds)
        loss = 10 * fp + fn + self.LOSS_FUNCTION(preds, truth)
        return loss

    @property
    def __name__(self):
        return "BERT_CNN_TORCH"

    def build_layers(self, embed_size):
        return BertCnnTorchModel(embed_size)
