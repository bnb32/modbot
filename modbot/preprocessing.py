"""Preprocessing methods"""
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

import re
import pkg_resources
from symspellpy import SymSpell
from collections import defaultdict
from tqdm import tqdm
from emoji import demojize
from googletrans import Translator
import dask.dataframe as dd
import copy
import pprint

from modbot.environment import ProcessingConfig
from modbot.utilities.utilities import (simple_chars_equal,
                                        remove_special_chars,
                                        is_user_type_chatty,
                                        delete_usernames,
                                        remove_reps,
                                        prune_chars,
                                        INFO_DEFAULT)
from modbot.utilities.logging import get_logger

stop_words = ENGLISH_STOP_WORDS
wordnet_lemmatizer = WordNetLemmatizer()
logger = get_logger()
translator = Translator()

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

sym_spell_seg = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell_seg.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


# Read in data
def read_data(dfile):
    """Read csv file and create DataFrame

    Parameters
    ----------
    dfile : str
        String for csv file path

    Returns
    -------
    texts : list
        List of texts to use for training
    y : list
        Corresponding list of targets
    """
    logger.info('Reading in data: %s', dfile)
    data = dd.read_csv(dfile)
    texts = data['text'].astype(str)
    y = data['is_offensive']
    return texts, y


def correct_msg(line):
    """Santitize message. Delete usernames, prune characters, and remove
    repetitions

    Parameters
    ----------
    line : str
        String containing message

    Returns
    -------
    str
        Sanitized message
    """
    tmp = line
    tmp = delete_usernames(tmp)
    tmp = prune_chars(tmp)
    tmp = remove_reps(tmp, sym_spell.words)
    return tmp


def segment_words(line):
    """Segment message into words

    Parameters
    ----------
    line : str
        String containing message

    Returns
    -------
    str
        Segmented line
    """
    if line != "":
        return sym_spell_seg.word_segmentation(line.strip()).segmented_string
    else:
        return line


def my_tokenizer(s):
    """Tokenize string

    Parameters
    ----------
    s : str
        String containing message

    Returns
    -------
    list
        List containing tokens
    """
    words = s.lower().split()
    words = [prune_chars(w) for w in words if w not in stop_words]
    return my_lemmatizer(words)


def remove_stopwords(words):
    """Remove stop words from word list

    Parameters
    ----------
    words : list
        List of words

    Returns
    -------
    list
        List of words with stop words removed
    """
    return [word for word in words if word not in stop_words]


def my_lemmatizer(words):
    """Lemmatize words

    Parameters
    ----------
    words : list
        List of words to lemmatize

    Returns
    -------
    list
        List of lemmatized words
    """
    return [wordnet_lemmatizer.lemmatize(w) for w in words]


def preproc_words(texts):
    """Tokenize texts

    Parameters
    ----------
    texts : list
        List of texts to tokenize

    Returns
    -------
    list
        List of token list for each text in texts
    """
    return [my_tokenizer(text) for text in texts]


def join_words(lines):
    """Join words back together after splitting

    Parameters
    ----------
    lines : list
        List of lines to join together

    Returns
    -------
    list
        List of joined lines
    """
    return [' '.join(line) for line in lines]


def filter_emotes(line, proc_config):
    """Filter emotes from a single line

    Parameters
    ----------
    line : str
        String containing message with emotes
    proc_config : ProcessingConfig
        Class storing preprocessing configuration

    Returns
    -------
    str
        String containing message without emotes
    """
    tmp = line
    for emote in proc_config.EMOTES:
        tmp = re.sub(emote, '', tmp, flags=re.I)
    return tmp.rstrip().lstrip()


def filter_all_emotes(texts, y, proc_config):
    """Filter emotes from all texts

    Parameters
    ----------
    texts : list
        List of texts to filter
    y : list
        List of corresponding targets
    proc_config : ProcessingConfig
        Class storing preprocessing configuration

    Returns
    -------
    texts : list
        List of filtered texts
    y : list
        List of corresponding targets
    """
    tmp_texts = []
    tmp_y = []
    for t, v in zip(texts, y):
        tmp = filter_emotes(t, proc_config)
        if tmp.strip() != '':
            tmp_texts.append(tmp)
            tmp_y.append(v)
    return tmp_texts, tmp_y


def write_data(outfile, texts, y):
    """Write data to outfile

    Parameters
    ----------
    outfile : str
        Path to outfile
    texts : list
        List of texts to write to file
    y : list
        List of targets to write to file
    """
    index = 0
    with open(outfile, 'w', encoding='utf-8') as f:
        logger.info(f"Writing data: {outfile}")
        f.write('index,is_offensive,text\n')
        for t, v in zip(texts, y):
            if re.sub('[^A-Za-z]+', '', t) != "":
                if int(v) == 1:
                    f.write(f'{index},{v},"{t}"\n')
                    index += 1
        for t, v in zip(texts, y):
            if re.sub('[^A-Za-z]+', '', t) != "":
                if int(v) == 0:
                    f.write(f'{index},{v},"{t}"\n')
                    index += 1


def check_msgs(line, checks):
    """Check line for specified phrases

    Parameters
    ----------
    line : str
        String containing message to check for phrases
    checks : list
        List of phrases to check

    Returns
    -------
    bool
    """
    for cp in checks:
        pattern = r'\b{}*\?*\b'.format(cp + cp[-1])
        if re.search(pattern, line, re.I):
            return True
    return False


def contains_link(text):
    """Check whether message contains a link

    Parameters
    ----------
    text : str
        String containing message to check for link
    proc_config : ProcessingConfig
        Class storing preprocessing configuration

    Returns
    -------
    bool
    """
    check = text is not None
    check = check and any(link in text for link
                          in ProcessingConfig.LINK_STRINGS)
    return check


def filter_log(infile, outfile, proc_config):
    """Filter log. Remove emotes and repetitions

    Parameters
    ----------
    infile : str
        Path to log file
    outfile : str
        Path to cleaned output file
    proc_config : ProccesingConfig
        Class storing processing configuration parameters
    """
    texts, y = read_data(infile)
    logger.info("Filtering Log")
    for n, t in tqdm(enumerate(texts)):
        tmp = filter_emotes(t, proc_config)
        tmp = re.sub('[^A-Za-z0-9 ]+', '', tmp)
        tmp = remove_reps(tmp, sym_spell.words)
        texts[n] = tmp
    write_data(outfile, texts, y)


def separate_links(infile, outfile):
    """Separate messages with and without links

    Parameters
    ----------
    infile : str
        Path to log file
    outfile : str
        Path to cleaned output file
    """
    texts, y = read_data(infile)
    links = []
    nolinks = []
    yl = []
    ynl = []
    for n, t in enumerate(texts):
        if contains_link(t):
            links.append(t)
            yl.append('1')
        else:
            nolinks.append(t)
            ynl.append(y[n])
    write_data(outfile, nolinks, ynl)
    write_data(outfile[0:-4] + '_links.csv', links, yl)


def correct_messages(infile, outfile):
    """Sanitize messages for easier classification

    Parameters
    ----------
    infile : str
        Path to log file
    outfile : str
        Path to cleaned output file
    """
    texts, y = read_data(infile)
    logger.info("Correcting messages")
    for n, t in tqdm(enumerate(texts)):
        texts[n] = correct_msg(t)
    write_data(outfile, texts, y)


class LogCleaning:
    """Class to handle different types of log cleaning"""
    def __init__(self, config, model=None, filter_emotes=False):
        """Initialize log cleaning

        Parameters
        ----------
        config : Config
            Config class with processing parameters
        model : ModerationModel
            Model class instance which can be used to check raw chat
            probabilities
        filter_emotes : bool
            Whether to filter emotes from raw chat data
        """
        self.config = ProcessingConfig(run_config=config)
        self.mem = MsgMemory()
        self.model = model
        self.review_decisions = getattr(config, 'review_decisions', False)
        self.filter_emotes = filter_emotes

        logger.info('Using processing configuration:\n'
                    f'{pprint.pformat(self.config.attrs, indent=1)}')

    @staticmethod
    def is_valid_line(line):
        """Check if line contains relevant information

        Parameters
        ----------
        line : str
            String containing line to check

        Returns
        -------
        bool
        """
        line_starts = ['BAN:', 'MOD_ACTION:', 'DELETED:']
        return line.startswith(tuple(line_starts + ['<']))

    @classmethod
    def read_log(cls, rawfile):
        """Read log and return lines

        Parameters
        ----------
        rawfile : str
            Path to raw chat file

        Returns
        -------
        list
            List of lines from log
        """
        # read raw log
        logger.info('Reading log: %s', rawfile)
        with open(rawfile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.lstrip() for line in tqdm(lines)]
        return lines

    @classmethod
    def prep_log(cls, rawfile):
        """Read log and do some preprocessing. Remove usernames and only
        return valid lines

        Parameters
        ----------
        rawfile : str
            Path to raw chat data

        Returns
        -------
        list
            List of valid lines from log with usernames removed
        """
        # prep log
        lines = cls.read_log(rawfile)
        logger.info('Removing usernames, escape chars, and non-valid lines')
        return [delete_usernames(
            re.sub('"', '\'', line)).rstrip('\n').rstrip().lstrip()
            for line in tqdm(lines) if cls.is_valid_line(line)]

    def review_messages(self, tocheck, bmsgs, cmsgs, ctmp, probs):
        """Perform initial review on temporarily clean messages

        Parameters
        ----------
        tocheck : list
            List of messages to manually check
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome
        ctmp : list
            Messages another layer of review
        probs : list
            List of probabilities for the messages in ctmp

        Returns
        -------
        tocheck : list
            List of messages to manually check
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome
        """
        for n in tqdm(range(len(ctmp))):
            text = ctmp[n]
            if self.model is not None and probs[n] > self.config.CHECK_PMIN:
                logger.info(f'Appending tocheck: {text}')
                tocheck.append(text)
            elif check_msgs(text.lower(), self.config.BLACKLIST):
                bmsgs.append(text)
            elif check_msgs(text.lower(), self.config.GRAYLIST):
                logger.info(f'Appending tocheck: {text}')
                tocheck.append(text)
            else:
                cmsgs.append(text)
        return tocheck, bmsgs, cmsgs

    def divide_messages(self, user, tocheck, bmsgs, cmsgs, ctmp):
        """Divide messages into clean, bad, and to_check

        Parameters
        ----------
        user : str
            User to check messages for
        tocheck : list
            List of messages to manually check
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome
        ctmp : list
            Messages another layer of review

        Returns
        -------
        tocheck : list
            List of messages to manually check
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome
        ctmp : list
            Messages another layer of review
        """
        for m in self.mem.chunks[user]:
            if m['msg'] is not None:
                to_check_append = (m['banned'] and m['mod']
                                   in self.config.BAN_CHECKS)
                to_check_append = to_check_append or (
                    m['deleted'] and m['mod'] in self.config.DELETE_CHECKS)
                to_check_append = to_check_append and (
                    m['mod'] not in self.config.IGNORE_ACTIONS)

                ban_append = (m['deleted'] or m['banned']
                              and m['mod'] not in self.config.IGNORE_ACTIONS
                              and m['mod'] is not None)
                clean_append = (not m['deleted'] and not m['banned'])

                if ((m['isSub'] is False or m['isSub'] is True)
                        and not contains_link(m['msg'])):

                    for w in self.config.WHITELIST:
                        m['msg'] = re.sub(w, '', m['msg'], flags=re.I)

                    if to_check_append:
                        logger.info('Appending tocheck (mod / msg): '
                                    f'{m["mod"]} / {m["msg"]}')
                        tocheck.append(m['msg'])
                    elif ban_append:
                        bmsgs.append(m['msg'])
                    elif clean_append:
                        ctmp.append(m['msg'])

        return tocheck, bmsgs, cmsgs, ctmp

    def further_review(self, tocheck, bmsgs, cmsgs):
        """Review tocheck messages and append to either cmsgs or bmsgs

        Parameters
        ----------
        tocheck : list
            List of messages to manually check
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome

        Returns
        -------
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome
        """

        logger.info("Reviewing to-check messages: %s", (len(tocheck)))
        for text in tqdm(tocheck):
            check_one = (check_msgs(text.lower(), self.config.BLACKLIST)
                         or remove_special_chars(text.lower()) in
                         (remove_special_chars(bmsg.lower())
                          for bmsg in bmsgs))
            check_zero = (remove_special_chars(text.lower()) in
                          (remove_special_chars(cmsg.lower())
                           for cmsg in cmsgs))
            if check_one:
                rating = '1'
            elif check_zero:
                rating = '0'
            else:
                trans_result = translator.translate(text)
                if trans_result.src != 'en':
                    rating = input(f'\nCheck: {text}\nTranslated: '
                                   f'{trans_result.text}\n')
                else:
                    rating = input(f'\nCheck: {text}\n')

            if rating not in ['0', '1', 's']:
                rating = '0'
            if rating == 's':
                logger.info("Skipped: %s", text)
            elif rating == '1':
                bmsgs.append(text)
            elif rating == '0':
                cmsgs.append(text)
        return bmsgs, cmsgs

    def append_messages(self, bmsgs, cmsgs):
        """Append messages to final texts array

        Parameters
        ----------
        bmsgs : list
            Messages classified as non-wholesome
        cmsgs : list
            Messages classified as wholesome

        Returns
        -------
        texts : list
            Final list of texts to write to file
        y : list
            Corresponding list of classifications/targets
        """

        texts = []
        y = []

        logger.info("Appending banned messages: %s", (len(bmsgs)))
        for text in tqdm(bmsgs):
            texts.append(text)
            y.append('1')

        logger.info("Appending clean messages: %s", (len(cmsgs)))
        for text in tqdm(cmsgs):
            texts.append(text)
            y.append('0')

        if self.filter_emotes:
            logger.info("Filtering emotes")
            texts, y = filter_all_emotes(texts, y, self.config)

        return texts, y

    def clean_log(self, rawfile, cleanfile=None):
        """Clean log. Do preprocessing and classify messages according to
        whether they were timed out or banned by a moderator. Manually check
        some lines if they contain certain phrases or if they were moderated
        by this bot.

        Parameters
        ----------
        rawfile : str
            Path to raw chat data
        cleanfile : str
            Path to clean output file
        """
        # prep log
        lines = self.prep_log(rawfile)
        self.mem.build_full_memory(lines)
        self.mem.chunk_memory()

        cmsgs = []
        bmsgs = []
        tocheck = []
        ctmp = []
        probs = []
        banned_user_count = 0
        clean_user_count = 0

        if self.review_decisions:
            logger.info("Preparing previous bot decisions for review")
            for user in tqdm(self.mem.chunks):
                for m in self.mem.chunks[user]:
                    if m['mod'] == self.config.NICKNAME:
                        tocheck.append(m['msg'])
        else:
            logger.info(
                "Dividing messages into 'bad' and 'other' for each user")
            for user in tqdm(self.mem.chunks):
                valid_check = (user not in self.config.IGNORE_USERS
                               and not self.mem.chunks[user][0]['isVip']
                               and not self.mem.chunks[user][0]['isMod']
                               and not self.mem.chunks[user][0]['isPartner'])
                if valid_check:
                    ban_check = any(m['banned'] for m in self.mem.chunks[user])
                    ban_check = ban_check or any(m['deleted'] for m
                                                 in self.mem.chunks[user])
                    if ban_check:
                        banned_user_count += 1
                    else:
                        clean_user_count += 1

                    out = self.divide_messages(user, tocheck, bmsgs, cmsgs,
                                               ctmp)
                    tocheck, bmsgs, cmsgs, ctmp = out

            logger.info("Banned/timed-out users: %s", banned_user_count)
            logger.info("Clean users: %s", clean_user_count)

            logger.info("Dividing 'other' into clean, to-check, and bad")
            if self.model is not None:
                logger.info("Calculating probabilities")
                probs = self.model.predict_one(ctmp)

            tocheck, bmsgs, cmsgs = self.review_messages(tocheck, bmsgs,
                                                         cmsgs, ctmp, probs)
            bmsgs, cmsgs = self.further_review(tocheck, bmsgs, cmsgs)
        texts, y = self.append_messages(bmsgs, cmsgs)
        if cleanfile is not None:
            write_data(cleanfile, texts, y)
        else:
            return texts, y


def get_info_from_chatty(line):
    """Populate info dictionary with info from IRC line

    Parameters
    ----------
    line : str
        String containing recent IRC message

    Returns
    -------
    dict
        Info dictionary
    """
    info = copy.deepcopy(INFO_DEFAULT)
    mod = None
    msg = None
    if line.startswith('<'):
        user = line.split('>')[0]
        user = user.strip('<')
        msg = line[line.index('>') + 1:].lstrip()
        info['isSub'] = is_user_type_chatty("sub", user)
        info['isMod'] = is_user_type_chatty("mod", user)
        info['isVip'] = is_user_type_chatty("vip", user)
        info['isPartner'] = is_user_type_chatty("partner", user)
        info['isPleb'] = is_user_type_chatty("pleb", user)
    elif line.startswith('BAN:'):
        user = line.split()[1].lower()
    elif line.startswith('DELETED:'):
        user = line.split()[1].lower()
        msg = ' '.join(line.split()[2:]).strip('(').strip(')')
    elif line.startswith('MOD_ACTION:'):
        user = line.split()[3].lower()
        mod = line.split()[1].lower()
        action = line.split()[2].lower().replace('(', '')
        if 'delete' in action:
            if line[-2:] == '))':
                msg = ' '.join(line.split()[4:]).strip('(').strip(')')
            else:
                msg = ' '.join(line.split()[4:-1]).strip('(').strip(')')
            info['deleted'] = True
        if 'ban' in action:
            info['banned'] = True
    info['line'] = line
    info['msg'] = info['raw_msg'] = msg
    info['mod'] = mod
    info['user'] = remove_special_chars(user).lower()
    return info


class MsgMemory:
    """Class to store messages and message info
    """
    def __init__(self):
        self.memory = defaultdict(list)
        self.chunks = defaultdict(list)
        self.msg_limit = 1

    def add_msg(self, info):
        """Add message to memory

        Parameters
        ----------
        info : dict
            Dictionary containing user info
        """
        entry = {k: v for k, v in info.items() if k != 'user'}
        if entry['msg'] is not None:
            entry['msg'] = demojize(entry['msg'])
        self.memory[info['user']].append(entry)

    def del_msg(self, user):
        """Remove message from memory dictionary

        Parameters
        ----------
        user : str
            Username to remove from memory
        """
        self.memory[user].pop(0)

    def check_msgs(self, user):
        """Check if user has messages in memory dictionary

        Parameters
        ----------
        user : str
            Username to check messages for
        """
        if len(self.memory[user]) > 0:
            if self.memory[user][-1]['banned']:
                self.clear_user(user)
            elif len(self.memory[user]) > self.msg_limit:
                self.del_msg(user)

    def update_user_ban(self, user, banned=True):
        """Update the ban status of the user

        Parameters
        ----------
        user : str
            Username to use for updating
        banned : bool
            Whether the user was banned or not
        """
        self.memory[user][-1]['banned'] = banned

    def clear_user(self, user):
        """Clear memory for user

        Parameters
        ----------
        user : str
            Username to clear memory for
        """
        self.memory[user] = []

    def update_banned_status(self, line):
        """Update banned status for user log

        Parameters
        ----------
        line : str
            Log line that could contain a user name
        """
        user = line.split()[1].lower()
        if user in self.memory and self.memory[user]:
            self.memory[user][-1]['banned'] = True

    def update_deleted_status(self, line):
        """Update deleted status for user log

        Parameters
        ----------
        line : str
            Log line that could contain a user name
        """
        info = get_info_from_chatty(line)
        user = info['user']
        if user not in self.memory or not self.memory[user]:
            self.add_msg(info)
        check = simple_chars_equal(info['raw_msg'],
                                   self.memory[user][-1]['raw_msg'])
        if not check:
            msg = ('Found conflicting delete action. Action message: '
                   f'{info["raw_msg"]}. Memory message: '
                   f'{self.memory[user][-1]["raw_msg"]}')
            logger.extra_verbose(msg)
            self.add_msg(info)
        self.memory[user][-1]['deleted'] = True

    def update_mod_action_status(self, line):
        """Update mod action status for user log

        Parameters
        ----------
        line : str
            Log line that could contain a user name
        """
        info = get_info_from_chatty(line)
        user = info['user']
        if user not in self.memory or not self.memory[user]:
            self.add_msg(info)
        if info['msg']:
            check = simple_chars_equal(info['raw_msg'],
                                       self.memory[user][-1]['raw_msg'])
            if not check:
                msg = ('Found conflicting mod action. Action message: '
                       f'{info["raw_msg"]}. Memory message: '
                       f'{self.memory[user][-1]["raw_msg"]}')
                logger.extra_verbose(msg)
                self.add_msg(info)
        self.memory[user][-1]['mod'] = info['mod']

    def build_full_memory(self, lines):
        """Build full memory for all lines

        Parameters
        ----------
        lines : list
            List of lines from which to build memory
        """
        logger.info("Building full memory")
        for line in tqdm(lines):
            action_check = ('(ban' in line or '(delete' in line
                            or '(timeout' in line)
            if line.startswith('<'):
                info = get_info_from_chatty(line)
                self.add_msg(info)
            elif line.startswith('BAN:'):
                self.update_banned_status(line)
            elif line.startswith('DELETED:'):
                self.update_deleted_status(line)
            elif line.startswith('MOD_ACTION:') and action_check:
                self.update_mod_action_status(line)

    def chunk_memory(self):
        """Chunk memory. Group messages by user and by chunk size so that
        more than one message can be taken into account for moderation.
        """
        logger.info('Chunking memory')
        info = None
        for user in tqdm(self.memory):
            count = 0
            for m in self.memory[user]:
                if m['msg']:
                    if count == 0:
                        info = copy.deepcopy(INFO_DEFAULT)

                    if info['msg']:
                        info['msg'] = m['msg'] + '. '
                    else:
                        info['msg'] += m['msg'] + '. '

                    for k in m:
                        if k not in ('msg', 'mod'):
                            info[k] = m[k]
                        if k == 'mod' and m['mod'] is not None:
                            info['mod'] = m['mod']

                    count += 1
                    count_check = (count == self.msg_limit
                                   or count == len(self.memory[user])
                                   or info['banned'] or info['deleted'])
                    if count_check:
                        entry = {k: v for k, v in info.items() if k != 'user'}
                        self.chunks[user].append(entry)
                        count = 0

    def chunk_recent(self, info):
        """Join number of past messages into a single string to use for
        model prediction

        Parameters
        ----------
        info : dict
            Dictionary containing user info

        Returns
        -------
        str
            String containing joined messages
        """
        self.check_msgs(info['user'])
        self.add_msg(info)
        msgs = [m['msg'].strip('\r') for m
                in self.memory[info['user']][-self.msg_limit:]]
        return '. '.join(msgs)


def separate_tocheck(config, infile, bounds, wc):
    """Separate to_check from all other messages

    Parameters
    ----------
    config : ProcessingConfig
        Class storing preprocessing configuration parameters
    infile : str
        Input file path
    bounds : list
        List with starting and ending message index to check
    wc : WholesomeCheck
        WholesomeCheck class to use for probability predictions

    Returns
    -------
    to_check : list
        List of messages to check
    y_check : list
        List of corresponding to check classifications
    texts : list
        List of final messages
    y : list
        List of correposponding final classifications
    """
    tmp_texts, tmp_y = read_data(infile)
    texts = []
    y = []
    to_check = []
    y_check = []

    logger.info("Separating to_check from all data")
    if wc is not None:
        logger.info("Calculating probabilities")
        probs = wc.predict_prob(tmp_texts)
    for n, t in tqdm(enumerate(tmp_texts)):
        if (wc is not None and bounds[0] <= n <= bounds[1] and tmp_y[n] == 0
                and not contains_link(t)
                and probs[n] > config.CHECK_PMIN):
            to_check.append(t)
            y_check.append(tmp_y[n])
        else:
            texts.append(t)
            y.append(tmp_y[n])
    return to_check, y_check, texts, y


def check_probs(config, infile, outfile, bounds, wc):
    """Check messages for reclassification if they have a non-wholesome
    probability above the minimum threshold

    Parameters
    ----------
    config : ProcessingConfig
        Class storing preprocessing configuration parameters
    infile : str
        Input file path
    outfile : str
        Output file path
    bounds : list
        List with starting and ending message index to check
    wc : WholesomeCheck
        WholesomeCheck class to use for probability predictions
    """

    to_check, y_check, texts, y = separate_tocheck(config, infile, bounds, wc)

    logger.info("Reclassifying to_check messages")
    end_check = False
    for n, t in tqdm(enumerate(to_check)):
        if not end_check:
            rating = input('\nCheck: %s, %s' % (y_check[n], t))
            if rating not in ['0', '1', 's', 'e']:
                rating = y_check[n]

            if rating == 's':
                logger.info("Skipped: %s", t)

            if rating == 'e':
                end_check = True

            if rating in ['0', '1']:
                texts.append(t)
                y.append(rating)
        else:
            for i in range(n, len(to_check)):
                texts.append(to_check[i])
                y.append(y_check[i])

    write_data(outfile, texts, y)


def check_phrases(infile, outfile):
    """Check phrases for reclassification

    Parameters
    ----------
    infile : str
        Input file path
    outfile : str
        Output file path
    """
    tmp_texts, tmp_y = read_data(infile)
    phrases = input('\nEnter phrases to check\n')
    phrases = phrases.split(',')
    texts = []
    y = []
    to_check = []
    y_check = []

    logger.info("Separating to_check from all data")
    for n, t in tqdm(enumerate(tmp_texts)):
        if check_msgs(t.lower(), phrases):
            to_check.append(t)
            y_check.append(tmp_y[n])
        else:
            texts.append(t)
            y.append(tmp_y[n])

    logger.info("Reclassifying to_check messages")
    end_check = False
    for n, t in tqdm(enumerate(to_check)):
        if not end_check:
            rating = input('\nCheck: %s, %s' % (y_check[n], t))
            if rating not in ['0', '1', 's', 'e']:
                rating = y_check[n]

            if rating == 's':
                logger.info("Skipped: %s", t)

            if rating == 'e':
                end_check = True

            if rating in ['0', '1']:
                texts.append(t)
                y.append(rating)
        else:
            for i in range(n, len(to_check)):
                texts.append(to_check[i])
                y.append(y_check[i])

    write_data(outfile, texts, y)
