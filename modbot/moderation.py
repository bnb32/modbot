"""Moderation module"""

from datetime import datetime as dt
from datetime import timedelta

import modbot.preprocessing as pp
from modbot.utilities import utilities
from modbot.environment import ProcessingConfig
from modbot.utilities.utilities import replies, UserInfo
from modbot.utilities.logging import get_logger
from modbot.training import get_model_class


logger = get_logger()


class Permitting:
    """Class to handle user permits - if they are exempt from moderation
    actions or can post links.
    """
    def __init__(self, run_config):
        self.config = run_config
        self.proc_config = ProcessingConfig(run_config.FILE_NAME)
        self.permits = {}
        self.permit_duration = 180

    def add_permit(self, user, length):
        """Add user to permit queue for a given length of time

        Parameters
        ----------
        user : str
            Username
        length : int
            Length of time in seconds
        """
        if user not in self.permits:
            self.permits[user] = {'start': dt.now(),
                                  'length': timedelta(seconds=length)}

    def del_permit(self, user):
        """Delete user from permit queue

        Parameters
        ----------
        user : str
            Username
        """
        if user in self.permits:
            self.permits.pop(user)

    def has_permit(self, user):
        """Check if user has a permit

        Parameters
        ----------
        user : str
            Username

        Returns
        -------
        bool
        """
        if user in self.permits:
            return True
        else:
            return False

    def time_permit(self, user):
        """Keep track of time a user is permitted and remove them if max
        time has elapsed

        Parameters
        ----------
        user : str
            Username to check for whether they have a permit
        """
        if self.has_permit(user):
            tmp = self.permits[user]['start'] + self.permits[user]['length']
            if dt.now() >= tmp:
                self.del_permit(user)

    @staticmethod
    def get_permit_user(msg):
        """Get user name from chat message for user to permit

        Parameters
        ----------
        msg : str
            Message to get username from

        Returns
        -------
        str
            Username
        """
        user = msg.split()
        user = user[1].lstrip('@')
        return user.lower()

    def get_permit_time(self, msg):
        """Get length of time to permit user from message. If message does not
        include a time then use default time.

        Parameters
        ----------
        msg : str
            Message to get permit time from

        Returns
        -------
        int
            Permit time in seconds
        """
        length = msg.split()
        try:
            return int(length[2])
        except Exception:
            return self.permit_duration

    def rep_permit(self, info):
        """Handle response to !permit command. Check if we should permit
        the user in info

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message

        Returns
        -------
        bool
        """
        if info['isMod'] and "!permit" in info['msg']:
            user = self.get_permit_user(info['msg'])
            length = self.get_permit_time(info['msg'])
            self.add_permit(user, length)
            return True
        else:
            self.time_permit(info['user'])
            return False


class Nuking:
    """Class to handle nuking words/phrases from chat. In development.
    """
    def __init__(self):
        self.nuke_phrase = None
        self.nuke_ongoing = False
        self.nuke_start = None
        self.nuke_end = None
        self.nuke_timeout = None
        self.initial_nuke_complete = False

    def rep_nuke(self, info):
        """Handle response to !nuke command. Check if we should send nuke.

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message

        Returns
        -------
        bool
        """
        if info['isMod'] and "!nuke" in info['msg']:
            self.nuke_ongoing = True
            self.initial_nuke_complete = False
            args = info['msg'].split()

            try:
                self.nuke_timeout = utilities.parse_time(args.pop(-1))
                scrollback_time = utilities.parse_time(args.pop(-1))
                self.nuke_start = dt.now() - scrollback_time
                if '-r' in args[-1]:
                    radiation_time = utilities.parse_radiation(args.pop(-1))
                    self.nuke_end = dt.now() + radiation_time
                else:
                    radiation_time = 0
                    self.nuke_end = dt.now()
                self.nuke_phrase = ' '.join(args[1:]).strip()
                start = dt.now() - timedelta(seconds=scrollback_time)
                tmp = f'Nuke start: {start}'
                end = dt.now() + timedelta(seconds=radiation_time)
                tmp += f'Nuke end: {end}'
                logger.mod(tmp)
                return True

            except Exception:
                return False

        else:
            self.time_nuke()
            return False

    def time_nuke(self):
        """End nuke if the nuke time has elapsed"""
        if self.nuke_ongoing and dt.now() > self.nuke_end:
            self.nuke_ongoing = False
        else:
            pass

    def should_nuke(self, msg):
        """Check if a nuke is ongoing and if there is a message in chat that
        needs to be nuked

        Parameters
        ----------
        msg : str
            Message to check for whether it should be nuked

        Returns
        -------
        bool
        """
        return (self.nuke_ongoing and self.nuke_phrase.lower() in msg.lower())


class Moderation(Permitting, Nuking):
    """Class to handle moderation actions and decisions
    """
    def __init__(self, run_config):
        Permitting.__init__(self, run_config)
        Nuking.__init__(self)
        self.model = self.initialize_model(run_config)
        self.run_config = run_config
        self.proc_config = ProcessingConfig(run_config.FILE_NAME)
        self.msgs = pp.MsgMemory()
        self.permits = {}
        self.pleb_prob = run_config.PLEB_PMIN
        self.sub_prob = run_config.SUB_PMIN
        self.send_prob_msg = run_config.PMSG_ON
        self.to_subs = run_config.TO_SUBS
        self.nolinks = run_config.NOLINKS
        self.timeout_duration = 5
        self.moderation_action = 'DELETE'
        self.words = self.proc_config.BLACKLIST

    @staticmethod
    def initialize_model(run_config):
        """Load trained model used for probability calculations

        Parameters
        ----------
        run_config : RunConfig
            Class storing run time configuration parameters

        Returns
        -------
        model : LSTM | SVM | CNN
        """
        MODEL_CLASS = get_model_class(run_config.MODEL_TYPE)
        model = MODEL_CLASS.load(run_config.MODEL_PATH, device='cpu')
        return model

    def filter_words(self, texts):
        """Check if words which need to be filtered are contained in texts

        Parameters
        ----------
        texts : list
            List of texts to check for filter words

        Returns
        -------
        bool
        """
        tmp = texts.lower()
        for w in self.words:
            if w in tmp:
                return True

    def get_info_prob(self, msg):
        """Get probability of non-wholesome message

        Parameters
        ----------
        msg : str
            Message to evaluate

        Returns
        -------
        float
        """
        tmp = pp.correct_msg(msg)
        tmp = self.remove_allowed_phrases(tmp)
        prob = round(self.model.predict_one([tmp])[0], 3)
        return prob

    @staticmethod
    def print_info(info):
        """Print info attributes to console

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message

        """
        if 'translated' in info:
            logger.chat(f"({info['badges']}) {info['user']}: "
                        f"{info['msg']}->{info['translated']} "
                        f"({info['prob']})")
        else:
            logger.chat(f"({info['badges']}) {info['user']}: {info['msg']} "
                        f"({info['prob']})")

    def remove_allowed_phrases(self, line):
        """Remove exempt phrases from line so probability of non-wholesome
        message is caculated without exempt phrases

        Parameters
        ----------
        line : str
            String containing most recent IRC message.
        """
        tmp = line.lower()
        for msg in self.proc_config.WHITELIST:
            if msg.lower() in tmp:
                tmp = tmp.replace(msg.lower(), '')
        return tmp

    def is_exempt(self, info):
        """Determine whether the user is exempt from moderation action

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message


        Returns
        -------
        bool
        """
        return bool(self.has_permit(info['user']) or info['isMod']
                    or info['isVip'] or info['isPartner']
                    or (info['isSub'] and not self.to_subs))

    def send_reply(self, stream_writer, info):
        """Make decision based on info and send moderation response

        Parameters
        ----------
        stream_writer : StreamWriter
            StreamWriter for IRC stream

        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message
        """
        # nuke response
        if self.rep_nuke(info):
            self.send_nuke(stream_writer)

        # permit response
        if self.rep_permit(info):
            user = self.get_permit_user(info['msg'])
            permit_time = self.get_permit_time(info['msg'])
            rep = replies['permit_msg'] % (user, permit_time)
            self.send_message(stream_writer, rep)
            logger.mod(f'Permitting {user} for {permit_time} seconds')

        # non-wholesome response
        elif self.send_prob_msg:
            if self.is_prob_exceeded(info):
                rep = replies['prob_msg'] % (info['user'], info['prob'])
                self.send_message(stream_writer, rep)

    @staticmethod
    def get_timeout(user, length, msg):
        """Get timeout string to use in moderation response

        Parameters
        ----------
        user : str
            Username
        length : int
            Length of timeout in seconds
        msg : str
            Message to include with timeout

        Returns
        -------
        str
            String containing timeout command
        """
        act = f"/timeout @{user} {length} {msg}"
        return act

    @staticmethod
    def get_delete(info):
        """Get delete string to use in moderation response

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message

        Returns
        -------
        str
            String containing delete command
        """
        act = f"/delete {info['msgId']}"
        return act

    def is_prob_exceeded(self, info):
        """Check is non-wholesome probability is exceeded for message in info

        Parameters
        ----------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message

        Returns
        -------
        bool
        """
        if info['isPleb']:
            return info['prob'] > self.pleb_prob
        if info['isSub']:
            return info['prob'] > self.sub_prob
        else:
            return False

    def send_nuke(self, stream_writer):
        """Send nuke to stream

        Parameters
        ----------
        stream_writer : StreamWriter
            StreamWriter for IRC stream
        """
        actions = {}
        if self.nuke_ongoing and not self.initial_nuke_complete:
            logger.mod(f'Launching nuke: {self.nuke_phrase}')
            for user in self.msgs.memory:
                for m in self.msgs.memory[user]:
                    if (self.nuke_phrase in m['msg']
                            and m['time'] > self.nuke_start):
                        rep = replies['nuke_comment'] % self.nuke_phrase
                        act = self.get_timeout(user, self.nuke_timeout, rep)
                        tmp_info = m.copy()
                        tmp_info['user'] = user
                        if not self.is_exempt(tmp_info):
                            actions[user] = act
                            m['banned'] = True
        self.initial_nuke_complete = True
        for _, action in actions.items():
            self.send_message(stream_writer, action)

    def send_action(self, stream_writer, info):
        """Send action to IRC stream

        Parameters
        ----------
        stream_writer : StreamWriter
            StreamWriter for IRC stream
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message
        """
        act = None

        # timeout
        if (self.is_prob_exceeded(info) and not self.is_exempt(info)):

            # update memory with ban
            self.msgs.update_user_ban(info['user'])

            if not self.send_prob_msg:
                rep = ''
            else:
                rep = replies['prob_msg'] % info['prob']

            if 'timeout' in self.moderation_action.lower():
                info['deleted'] = True
                act = self.get_timeout(info['user'],
                                       self.timeout_duration, rep)
            elif 'delete' in self.moderation_action.lower():
                info['banned'] = True
                act = self.get_delete(info)

        elif (pp.contains_link(info['msg']) and self.nolinks
              and not self.is_exempt(info)):

            rep = replies['link_msg']

            if 'timeout' in self.moderation_action.lower():
                info['deleted'] = True
                act = self.get_timeout(info['user'],
                                       self.timeout_duration, rep)
            elif 'delete' in self.moderation_action.lower():
                info['banned'] = True
                act = self.get_delete(info)

        elif self.should_nuke(info['msg']) and not self.is_exempt(info):
            info['banned'] = True
            rep = replies['nuke_comment'] % self.nuke_phrase
            act = self.get_timeout(info['user'], self.nuke_timeout, rep)

        self.send_message(stream_writer, act, info)

    def get_info_from_irc(self, line):
        """Populate info dictionary using info from line

        Parameters
        ----------
        line : str
            String containing most recent IRC message.

        Returns
        -------
        info : dict
            dictionary storing attributes of user. e.g. recent message, badges,
            username, probability of non-wholesome message
        """
        info = UserInfo.get_info(line)
        info['mod'] = self.run_config.NICKNAME
        info['msg'] = self.msgs.chunk_recent(info)
        info['prob'] = self.get_info_prob(info['msg'])
        return info

    def send_message(self, stream_writer, message=None, info=None):
        """Send message to IRC stream

        Parameters
        ----------
        stream_writer : StreamWriter
            StreamWriter for IRC stream
        message : str, optional
            Message string, by default None
        info : dict, optional
            Info dictionary, by default None
        """
        if message is not None:
            message_temp = f"PRIVMSG #{self.run_config.CHANNEL} :{message}"
            stream_writer.write(message_temp)
            if info is None:
                logger.mod(f"Sent: {message_temp}")
            else:
                logger.mod(
                    f"Sent: {message_temp}, {info['user']}: {info['msg']}")
