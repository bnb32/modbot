"""Initialize configuration parameters"""
import json
import os
from datetime import datetime
import warnings

from modbot import DATA_DIR as default_data_dir
from modbot import LOG_DIR as default_log_dir
from modbot import BERT_PREPROCESS as default_bert_preprocess
from modbot import BERT_ENCODER as default_bert_encoder


def get_model_path(model_type):
    """Get default model path

    Parameters
    ----------
    model_type : str
        Valid model type

    Returns
    -------
    model_path : str
        Path to save/load model
    """
    model_path = os.path.join(default_data_dir, f'{model_type.upper()}_model')
    if model_type.upper() in ('SVM'):
        model_path = os.path.join(model_path, 'model.joblib')
    return model_path


class BaseConfig:
    """Base config class"""

    #: Oauth token for IRC connection
    _TOKEN = None

    #: Client ID for PubSub connection
    _CLIENT_ID = None

    #: Client secret for PubSub connection
    _CLIENT_SECRET = None

    #: Bot username
    NICKNAME = None

    #: Channel to moderate
    CHANNEL = None

    #: Threshold probability for pleb moderation
    PLEB_PMIN = 0.6

    #: Threshold probability for sub moderation
    SUB_PMIN = 0.6

    #: Whether to send message in chat when probability is exceeded
    PMSG_ON = False

    #: Whether to moderate subscribers
    TO_SUBS = True

    #: Log level. 18 is Verbose
    LOGGER_LEVEL = 18

    #: Whether to write chat messages to log file
    WRITE_LOG = True

    #: Whether to allow links in chat or not
    NOLINKS = False

    #: Path to ML model
    MODEL_PATH = None

    #: Type of model to use for moderation
    MODEL_TYPE = 'SVM'

    #: Threshold probability to check for new classification
    CHECK_PMIN = 0.4

    #: Ratio of ones to total number of samples for training data
    OFFENSIVE_WEIGHT = 0.01

    #: Path to Chatty data for training
    CHATTY_DIR = None

    #: Directory where training data and model is stored
    DATA_DIR = default_data_dir

    #: Directory to write logs
    LOG_DIR = default_log_dir

    #: File to use for logging chat and mod actions
    LOG_PATH = None

    #: Path to bert preprocess model
    BERT_PREPROCESS = default_bert_preprocess

    #: Path to bert encoder
    BERT_ENCODER = default_bert_encoder

    def __init__(self, file_name=None, args=None):
        """
        Parameters
        ----------
        file_name : str
            JSON configuration file
        args : parser.parse_args
            Args from argparse method
        """
        check = (file_name is not None or args is not None)
        msg = ('Received neither config file path or args object. Using '
               'default configuration parameters.')
        if not check:
            warnings.warn(msg)

        if file_name is not None:
            self.file_name = file_name
        elif args is not None:
            self.file_name = args.config
        else:
            self.file_name = None
        self.config_dict = {}
        self.get_config(self.file_name)
        if args is not None:
            self.update_config(args)

    @property
    def attrs(self):
        """Config attributes info"""
        config_attrs = {k: getattr(self, k) for k in vars(self)}
        config_attrs.pop('config_dict', None)
        return config_attrs

    def get_log_file(self):
        """Get default log file name"""
        date_string = datetime.now().strftime("%Y-%m-%d")
        if self.LOG_PATH is None:
            self.LOG_PATH = os.path.join(self.LOG_DIR,
                                         f'{date_string}_#{self.CHANNEL}.log')

    def get_config(self, file_name=None, config=None):
        """Get configuration from file

        Parameters
        ----------
        file_name : str
            Path to configuration file
        config : RunConfig
            Optional config class with stored parameters

        Returns
        -------
        RunConfig
            Config class with run time configuration stored in attributes
        """

        if file_name is not None:
            with open(file_name, 'r') as fh:
                self.config_dict = json.load(fh)

            for k, v in self.config_dict.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        elif config is not None:
            for k in vars(config):
                if hasattr(self, k):
                    setattr(self, k, getattr(config, k))
        self.get_log_file()

    @property
    def public_attrs(self):
        """Get public attributes

        Returns
        -------
        dict
            Dictionary of public global attributes
        """
        config_attrs = {k: getattr(self, k) for k in self.attrs
                        if not k.startswith('_')}
        return config_attrs

    @property
    def upper_attrs(self):
        """Get public global attributes

        Returns
        -------
        dict
            Dictionary of public global attributes
        """
        config_attrs = {k: getattr(self, k) for k in self.attrs
                        if not k.startswith('_') and k.upper() == k}
        return config_attrs

    def update_config(self, args):
        """Update config with args

        Parameters
        ----------
        args : parser.parse_args
            Args from argparser with which to update config
        """

        args.model_type = (args.model_type if args.model_type is not None
                           else self.MODEL_TYPE)
        args.model_path = (get_model_path(args.model_type)
                           if args.model_path is None else args.model_path)
        for key in vars(args):
            val = getattr(args, key)
            if not hasattr(self, key):
                setattr(self, key, val)
            if hasattr(self, key.upper()) and getattr(args, key) is not None:
                setattr(self, key.upper(), val)
        self.get_log_file()

    def save(self, outpath):
        """Save config to json file

        Parameters
        ----------
        outpath : str
            Path to save public attrs
        """

        if os.path.isdir(outpath):
            model_dir = outpath
        else:
            model_dir = os.path.dirname(outpath)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as fh:
            params = dict(self.public_attrs)
            json.dump(params, fh)


class RunConfig(BaseConfig):
    """Config class to hold run time config parameters"""


class ProcessingConfig(BaseConfig):
    """Configuration for preprocessing routines"""

    #: Strings to indicate if message contains a link
    LINK_STRINGS = ['https://', 'http://', 'www.', '.com', '.net',
                    '.org', '.edu', '/tv', '.tv']

    #: Phrase lists for filtering and manual checking signals
    BLACKLIST = ['finger her', 'N I G G E R', '#removethemole',
                 'super hottie', 'thicc', 't h i c c', 'u are hot',
                 'you are hot', 'your hot', 'ur hot', 'u are sexy',
                 'you are sexy', 'your sexy', 'ur sexy',
                 'why is this in just chatting', 'change category',
                 'u a virgin', 'wrong category', 'si1mp', 's1mp',
                 'simp', 'si(.*)mp', 'T H I C C', 'only fans',
                 'onlyfans', 'thicc(.*)Botez', 'hot(.*) Botez',
                 'hot(.*) Andrea', 'hot(.*)Alex', 'thicc(.*)Alex',
                 'thicc(.*) andrea', 'Botez(.*) thicc',
                 'Botez(.*) hot', 'Andrea(.*) hot', 'Alex(.*) hot',
                 'Alex(.*)thicc', 'andrea(.*) thicc', 'she(.*)thicc',
                 'sexy(.*) Alex', 'sexy(.*) andrea', 'Botez(.*) sexy',
                 'sexy(.*) Botez', 'Andrea(.*)sexy', 'Alex(.*) sexy',
                 'andrea(.*) sexy', 'ur so hawt', 'u so hawt', 'rape',
                 'your butthole', 'mybutthole', 'give me a kiss',
                 'gimme a kiss', 'blow me a kiss', 'whore', 'pussy',
                 'cunt', 'suck dick', 'lick(.*) feet', 'finger you',
                 'suck my', 'whore', 'simp', 'lick(.*) toes',
                 'suck(.*) toes', 'your vagina', 'your vag',
                 'show vag', 'vagene', 'show bobs', 'vagoo',
                 'your booty', 'ur so hot', 'u so hot', 'slut']

    #: Phrases to check
    GRAYLIST = []

    #: Allowed phrases
    WHITELIST = ['LUL', 'KEKW', '!drama']

    #: List of emotes to use for filtering
    EMOTES = ['NotLikeThis', '<3', 'LUL', 'WutFace',
              'ResidentSleeper', 'pepeLa', 'pepeDS', 'pepeJAM',
              'pepePls', 'FeelsBadMan', 'FeelsGoodMan', 'monkaS',
              'monkaHmm', 'KEKW', 'Kapp', 'Hypers', 'HandsUp',
              'FeelsWeirdMan', 'EZY', '5Head', '3Lass', 'Pog',
              'Pepega', 'PagChomp', 'OMEGALUL', 'monkaW', 'PogU',
              'PepeHands', 'PepoThink', 'alexandraSub', 'PepePls',
              'PixelBob', 'Jebaited', 'TriHard', 'MrDestructoid',
              'Keepo', 'HeyGuys', 'PepeLaugh', 'SquadP', 'PJSugar',
              'ammoHeman', 'SourPls', 'peepoHappy', 'HotPokket',
              'BibleThump', 'DansGame', 'orbThonk', 'malexaH',
              'electraSad', 'TopHat', 'FutureMan', 'PokGengar',
              'OMEGA', 'onetri2Hii', 'bayliunSad', 'bayliunVictory',
              'cheer100', 'cheer10', 'cheer1', 'TBAngel',
              'chesscoachHi', 'chesscoachWhat', 'TwitchUnity',
              'justbu8', 'blinkxXmas', 'symfNerd', 'malexaHi',
              'HahaTurtledove', 'HahaThisisfine', 'HahaThink',
              'HahaSweat', 'HahaSleep', 'HahaShrugRight',
              'HahaShrugMiddle', 'HahaLean', 'HahaNutcracker',
              'HahaPoint', 'HahaSnowhal', 'HahaPresent',
              'HahaReindeer', 'HahaShrugLeft', 'HahaHide',
              'HahaGoose', 'HahaGingercat', 'HahaElf', 'HahaDoge',
              'HahaCat', 'HahaBaby', 'HahaNyandeer', 'Haha2020',
              'HahaBall', 'HahaDreidel', 'gutBaby']

    def __init__(self, file_name=None, run_config=None):
        """
        Parameters
        ----------
        file_name : str
            JSON configuration file
        run_config : RunConfig
            Optional config class with stored parameters
        """
        if run_config is not None:
            self.get_config(config=run_config)
        else:
            self.get_config(file_name=file_name)
            run_config = RunConfig(file_name=file_name)
        self.file_name = file_name

        #: ignore actions of these moderators during training
        self.ignore_actions = ['moobot', run_config.NICKNAME]

        #: review bans of these moderators during training
        self.ban_checks = [run_config.NICKNAME]

        #: review deletions by these moderators during training
        self.delete_checks = [run_config.NICKNAME]

        #: ignore messages of these users during training
        self.ignore_users = []

    @property
    def attrs(self):
        """Config attributes info"""
        config_attrs = {k: getattr(self, k)
                        for k in vars(self) if k == k.lower()}
        return config_attrs
