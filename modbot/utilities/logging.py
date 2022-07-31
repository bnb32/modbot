"""Logging module for storing chat and connection info"""
from emoji import demojize
import logging
from sys import stdout

from modbot.utilities.utilities import is_user_type_irc


VERBOSE_LEVEL = logging.INFO - 2
EXTRA_VERBOSE_LEVEL = logging.INFO - 4
CHAT_LEVEL = logging.INFO + 2
MOD_LEVEL = logging.INFO + 4
PRIVATE_LEVEL = logging.INFO - 10

COLORS = {
    'VERBOSE': [226, 226, 226],
    'EXTRA_VERBOSE': [202, 202, 202],
    'ERROR': [196, 196, 196],
    'MOD': [46, 46, 46],
    'CHAT': [33, 33, 33, 161, 135, 196]
}


class ColoredFormatter(logging.Formatter):
    """Colored formatting for log output"""
    level_format = '[%(levelname)s] '
    other_format = '%(filename)s:%(lineno)d %(asctime)s '
    msg_format = '%(message)s'
    full_format = level_format + other_format + msg_format

    COLORS = {
        'VERBOSE': [226, 226, 226],
        'EXTRA_VERBOSE': [202, 202, 202],
        'ERROR': [196, 196, 196],
        'MOD': [46, 46, 46],
        'CHAT': [33, 33, 33, 161, 135, 196]
    }

    FORMATS = {}
    for level in COLORS:
        COL_SEQ = f"\u001b[38;5;{COLORS[level][0]}m"
        val = COL_SEQ + level_format + "\u001b[0m"
        COL_SEQ = f"\u001b[38;5;{COLORS[level][1]}m"
        val += COL_SEQ + other_format + "\u001b[0m"
        COL_SEQ = f"\u001b[38;5;{COLORS[level][2]}m"
        val += COL_SEQ + msg_format + "\u001b[0m"
        FORMATS[level] = val

    def format(self, record):
        """Apply level specific coloring"""
        formatter = logging.Formatter(self.FORMATS.get(record.levelname,
                                                       self.full_format))
        if record.levelname == 'CHAT':
            msg = record.msg
            msg_split = msg.split()
            badges = msg_split[0]
            username = msg_split[1]
            msg = " ".join(msg_split[2:-1])
            prob = msg_split[-1]
            COL_SEQ = f"\u001b[38;5;{self.COLORS['CHAT'][-1]}m"
            prob = COL_SEQ + prob + "\u001b[0m"
            COL_SEQ = f"\u001b[38;5;{self.COLORS['CHAT'][-2]}m"
            msg = COL_SEQ + msg + "\u001b[0m"
            COL_SEQ = f"\u001b[38;5;{self.COLORS['CHAT'][-3]}m"
            username = COL_SEQ + username + "\u001b[0m"
            msg = " ".join([badges, username, msg, prob])
            record.msg = msg

        return formatter.format(record)


class CustomLogger(logging.getLoggerClass()):

    LEVELS = {'VERBOSE': logging.INFO - 2,
              'EXTRA_VERBOSE': logging.INFO - 4,
              'CHAT': logging.INFO + 2,
              'MOD': logging.INFO + 4,
              'PRIVATE': logging.INFO - 10}

    def __init__(self, name='modbot_logger', level=18):
        """Initialize logger

        Returns
        -------
        logger
            Logger object
        """
        super().__init__(name=name, level=level)

        if not self.handlers:
            for level in self.LEVELS:
                logging.addLevelName(self.LEVELS[level], level)

            sh = logging.StreamHandler(stdout)
            formatter = ColoredFormatter()
            sh.setFormatter(formatter)
            sh.setLevel(level)
            self.addHandler(sh)
            self.setLevel(level)
            self.propagate = False

    def extra_verbose(self, message, *args, **kws):
        """Extra verbose log level"""
        self._log(self.LEVELS['EXTRA_VERBOSE'], message, args, **kws)

    def verbose(self, message, *args, **kws):
        """Verbose log level"""
        self._log(self.LEVELS['VERBOSE'], message, args, **kws)

    def chat(self, message, *args, **kws):
        """Chat log level"""
        self._log(self.LEVELS['CHAT'], message, args, **kws)

    def mod(self, message, *args, **kws):
        """MOD log level"""
        self._log(self.LEVELS['MOD'], message, args, **kws)

    def private(self, message, *args, **kws):
        """MOD log level"""
        self._log(self.LEVELS['PRIVATE'], message, args, **kws)

    def update_level(self, level):
        """Update logger level

        Parameters
        ----------
        logger : modbot_logger
            Logger object
        level : str
            New level. e.g. VERBOSE
        """
        for sh in self.handlers:
            if level in self.LEVELS:
                levno = self.LEVELS[level]
            else:
                levno = getattr(logging, level, 10)
            sh.setLevel(levno)
        self.setLevel(levno)


def get_logger(name='modbot_logger', level=18):
    """Get main logger"""
    return CustomLogger(name=name, level=level)


class PrivateLogger(CustomLogger):

    def __init__(self, name='private_logger', level=10, log_path=None):
        """Initialize logger

        Returns
        -------
        logger
            Logger object
        """
        logging.getLoggerClass().__init__(self, name=name, level=level)

        if not self.handlers:
            for level in self.LEVELS:
                logging.addLevelName(self.LEVELS[level], level)
            logFormatter = logging.Formatter("%(message)s")
            fileHandler = logging.FileHandler(log_path)
            fileHandler.setFormatter(logFormatter)
            self.addHandler(fileHandler)
            self.propagate = False


class Logging:
    """Log handling class"""
    def __init__(self, run_config):
        self.log_path = run_config.LOG_PATH
        self.write_log = run_config.WRITE_LOG
        self.logger = None
        if self.write_log:
            self.logger = self.initialize_logger(run_config)
        self.UserLog = {}

    def initialize_logger(self, run_config):
        """Initialize private logger

        Parameters
        ----------
        run_config : RunConfig
            RunConfig class storing run time configuration parameters

        Returns
        -------
        logger
            Private logger used to write chat messages and mod actions
        """
        private_logger = PrivateLogger('private_logger',
                                       log_path=run_config.LOG_PATH)
        private_logger.update_level('PRIVATE')
        return private_logger

    @staticmethod
    def get_value(key, dictionary):
        """Search nested dictionary for key

        Parameters
        ----------
        key : str
            Key to search dictionary for
        dictionary : dict
            Dictionary to search key for

        Returns
        -------
        str
            Value in dictionary corresponding to key, or empty string if not
            found
        """
        if key in dictionary:
            return dictionary[key]
        else:
            for k in dictionary:
                if key in k:
                    return dictionary[k]
            return ''

    def append_log(self, line):
        """Add entry to log

        Parameters
        ----------
        line : str
            Line to write to log after some sanitizing
        """
        write_check = (not is_user_type_irc('vip', line)
                       and not is_user_type_irc('mod', line)
                       and self.write_log)
        if write_check:
            self.logger.private(line.encode('ascii', 'ignore').decode())

    @staticmethod
    def build_chat_log_entry(info):
        """Build standard chat message log entry

        Parameters
        ----------
        info : dict
            Dictionary of info from IRC message

        Returns
        -------
        str
            Chat log entry to write to log
        """
        log_entry = "<"
        if info['isMod']:
            log_entry += "@"
        if info['isVip']:
            log_entry += "!"
        if info['isSub']:
            log_entry += "%"
        if info['isPartner']:
            log_entry += "~"
        log_entry += info['user'] + "> "
        log_entry += info['msg']

        return log_entry

    @staticmethod
    def build_action_log_entry(action, user, moderator, msg, secs, msg_id):
        """Build action log entry from message content

        Parameters
        ----------
        action : str
            Moderation action. e.g ban, timeout, delete
        user : str
            Username
        moderator : str
            Moderator username who performed action
        msg : str
            The message which preceeded the mod action
        secs : str
            Timeout length in seconds
        msg_id : str
            Message id

        Returns
        -------
        str
            Log entry containing moderation action info
        """
        log_entry = ""
        if "ban" in action:
            log_entry = f"BAN: {user}\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user} {msg})"
        elif "timeout" in action:
            log_entry = f"BAN: {user} ({secs}s)\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user} {secs} {msg})"
        elif "delete" in action:
            log_entry = f"DELETED: {user} ({msg})\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user}"
            log_entry += f" {msg} {msg_id})"
        else:
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user})"
        return log_entry

    def get_info_from_pubsub(self, msg_dict):
        """Get info from PubSub message for logging

        Parameters
        ----------
        msg_dict : dict
            Dictionary containing PubSub message info

        Returns
        -------
        action : str
            Moderation action. e.g ban, timeout, delete
        user : str
            Username
        moderator : str
            Moderator username who performed action
        msg : str
            The message which preceeded the mod action
        secs : str
            Timeout length in seconds
        msg_id : str
            Message id
        """
        action = self.get_value('moderation_action', msg_dict)
        moderator = self.get_value('created_by', msg_dict)
        user = msg = secs = msg_id = ''
        if 'args' in msg_dict:
            user = msg_dict['args'][0]
        else:
            user = self.get_value('target_user_login', msg_dict)

        if action in ['ban', 'timeout']:
            if user in self.UserLog:
                msg = self.UserLog[user]
            else:
                msg = ''
        if "timeout" in action:
            secs = msg_dict['args'][1]
        if "delete" in action:
            msg = demojize(msg_dict['args'][1])
            msg_id = msg_dict['args'][2]
        return action, user, moderator, msg, secs, msg_id
