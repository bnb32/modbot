"""Custom vectorizers"""
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
import pickle

from modbot.utilities.logging import get_logger

logger = get_logger()


class TokVectorizer:
    """Tokenizer vectorizer for LSTM primarily"""
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250

    def __init__(self, tokenizer=None, max_len=None):
        if max_len is None:
            max_len = self.MAX_SEQUENCE_LENGTH
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        if tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS,
                                       char_level=True)
            logger.info('Successfully built vectorizer')
        else:
            logger.info('Successfully loaded vectorizer')
            self.tokenizer = tokenizer

    def tokenize(self, X):
        """Tokenize texts"""
        sequences = self.tokenizer.texts_to_sequences(X)
        sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_len)
        return sequences_matrix

    def encode(self, Y):
        """Encode target labels"""
        Y = self.label_encoder.fit_transform(Y)
        Y = Y.reshape(-1, 1)
        return Y

    def transform(self, X):
        """Transform texts"""
        X_out = self.tokenize(X)
        return X_out

    def fit(self, X):
        """Fit tokenizer on texts"""
        self.tokenizer.fit_on_texts(X)

    def fit_transform(self, X):
        """Fit transform method to conform to sklearn model format"""
        self.fit(X)
        return self.transform(X)

    def save(self, outpath):
        """Save tokenizer"""
        with open(outpath, 'wb') as fh:
            logger.info(f'Saving vectorizer to {outpath}')
            pickle.dump(self.tokenizer, fh)

    @classmethod
    def load(cls, inpath):
        """Load tokenizer"""
        tokenizer = joblib.load(inpath)
        tokenizer = cls(tokenizer=tokenizer)
        return tokenizer
