
"""WholesomeBot utilities"""
import re
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone
import copy
import time

replies = {'timeout': "",
           'bio': "/me I'm an AI developed by drchessgremlin, "
                  "learning what is wholesome and non-wholesome based on what"
                  " you write in chat.",
           'permit_msg': "/me @%s will not be timed out for %s seconds",
           'nuke_msg': "/me Nuking %s",
           'nuke_comment': "nuking %s",
           'prob_msg': "/me %s: non-wholesome p=%s",
           'prob_comment': "non-wholesome p=%s",
           'link_msg': "/me Please ask a moderator before posting a link"}

INFO_DEFAULT = {
    'line': None,
    'user': None,
    'msg': None,
    'isSub': False,
    'isMod': False,
    'isVip': False,
    'isPartner': False,
    'isPleb': False,
    'badges': None,
    'msgId': None,
    'role': None,
    'time': None,
    'deleted': False,
    'banned': False,
    'mod': None,
    'prob': 0.0,
}


def curvature(a):
    """Calculate curvature of a vector"""
    b = [0] * len(a)
    for i, _ in enumerate(b):
        if 0 < i < len(a) - 1:
            b[i] = (a[i + 1] - 2 * a[i] + a[i - 1])
        elif i == 0:
            b[i] = (a[i + 1] - 2 * a[i])
        elif i == len(a) - 1:
            b[i] = (-2 * a[i] + a[i - 1])
    return b


def none_or_str(arg):
    """Get none or string"""
    if arg == 'None':
        return None
    return arg


def none_or_int(arg):
    """Get none or int"""
    if arg == 'None' or arg is None:
        return None
    else:
        return int(arg)


def none_or_float(arg):
    """Get none or float"""
    if arg == 'None' or arg is None:
        return None
    else:
        return float(arg)


class UserInfo:
    """Class storing user information extracted from chat message"""

    @staticmethod
    def get_info(line):
        """
        Returns
        -------
        dict
            Default info dictionary
        """
        info = copy.deepcopy(INFO_DEFAULT)
        info['line'] = line
        info['user'] = get_user(line)
        info['msg'] = get_message(line)
        info['isSub'] = is_user_type_irc("sub", line)
        info['isMod'] = is_user_type_irc("mod", line)
        info['isVip'] = is_user_type_irc("vip", line)
        info['isPartner'] = is_user_type_irc("partner", line)
        info['isPleb'] = is_user_type_irc("pleb", line)
        info['badges'] = get_badges(line)
        info['msgId'] = get_msg_id(line)
        info['role'] = get_role(line)
        info['time'] = dt.now()
        info['deleted'] = False
        info['banned'] = False
        return info


def delete_usernames(line):
    """Remove usernames from messages

    Parameters
    ----------
    line : str
        Line containing message with usernames

    Returns
    -------
    str
        String with usernames removed
    """
    msg = re.sub(r'@\w+', '', line)
    return msg


def remove_reps(line, word_list):
    """Remove repetitions from messages"""
    p = re.compile(r'(.)\1+')
    words = p.sub(r'\1\1', line.lower()).split()
    for n, w in enumerate(words):
        if re.sub('[^A-Za-z0-9]+', '', w) not in word_list:
            tmp = p.sub(r'\1', w)
            if re.sub('[^A-Za-z0-9]+', '', tmp) in word_list:
                words[n] = tmp
    return ' '.join(words)


def date_time():
    """Get current date string

    Parameters
    ----------
    datetime
        String containing current datetime
    """
    date_str = dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    datetime_obj = dt.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    datetime_obj_pst = datetime_obj.replace(
        tzinfo=timezone('US/Eastern')) - timedelta(hours=3)
    return datetime_obj_pst.strftime("%Y-%m-%d %H:%M:%S.%f")


def parse_time(msg):
    """Get time from nuke command message

    Parameters
    ----------
    msg : str
        Nuke command message

    Returns
    -------
    int
        Time in seconds
    """
    if 'h' in msg:
        return int(msg.replace('h', '')) * 60 * 60
    elif 'm' in msg:
        return int(msg.replace('m', '')) * 60
    elif 's' in msg:
        return int(msg.replace('s', ''))
    else:
        return int(msg)


def parse_radiation(msg):
    """Get radiation time from nuke command message. This specifies how long
    ago to check messages for possible nuke.

    Parameters
    ----------
    msg : str
        Message including nuke command and nuke options

    Returns
    -------
    int
        Time of radiation in seconds
    """
    tmp = msg.split('-r=')
    return parse_time(tmp[1])


def remove_special_chars(line):
    """Remove all special characters from line

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        Line with all special characters removed
    """
    return re.sub('[^A-Za-z0-9_ ]+', '', line.rstrip('\n').rstrip().lstrip())


def simple_chars_equal(line1, line2):
    """Compare characters in line1 and line2

    Parameters
    ----------
    line1 : str
        First line to compare
    line2 : str
        Second line to compare
    """
    tmp1 = remove_special_chars(line1).replace(' ', '')
    tmp2 = remove_special_chars(line2).replace(' ', '')
    return tmp1 == tmp2


def prune_chars(line):
    """Remove some special characters from line

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        Line with some special characters removed
    """
    return re.sub('[^A-Za-z0-9_?#*!/-@ ]+', '',
                  line.rstrip('\n').rstrip().lstrip())


def get_line_type(line):
    """Get type of line from prefix

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        Type of line
    """

    type_map = {"CLEARCHAT": "ban",
                "CLEARMSG": "delete",
                "JOIN": "join",
                "PART": "part",
                "PRIVMSG": "msg"}
    for k, v in type_map.items():
        if k in line:
            return v
    return "misc"


def is_line_type(line_type, line):
    """Check if line contains the requested signal

    Parameters
    ----------
    line_type : str
        Signal type to check. e.g. msg, deleted, ban, etc
    line : str
        String containing recent IRC message

    Returns
    -------
    bool
    """
    return get_line_type(line) == line_type


def get_user(line):
    """Get user from line

    Parameters
    ----------
    line : str
        String containing recent IRC message

    Returns
    -------
    str
        Username
    """
    user = ''
    if is_line_type("msg", line):
        for line in line.split(';'):
            if line.startswith('user-type='):
                msg = re.search('user-type=(.*)', line).group(1)
                user = msg.split(":", 2)[1].split("!", 1)[0]
    elif is_line_type("delete", line):
        line = line.split(';')
        user = line[0].strip('@login=')
        user = user.replace('\r', '').replace('\n', '')
    elif is_line_type("ban", line):
        line = line.split(':')
        user = line[-1].strip()
        user = user.replace('\r', '').replace('\n', '')
    return user


def get_user_type_chatty(user):
    """Get user type list from symbols preceeding username

    Parameters
    ----------
    user : str
        Username string including symbols mapping to badges

    Returns
    -------
    list
        List of user types
    """
    user_type_map = {"%": "sub",
                     "@": "mod",
                     "!": "vip",
                     "~": "partner"}
    type_list = []
    for k, v in user_type_map.items():
        if k in user:
            type_list.append(v)
    return ["pleb"] if not type_list else type_list


def is_user_type_chatty(user_type, user):
    """Determine whether user is a specific type

    Parameters
    ----------
    user_type : str
        User type string. e.g. sub, pleb, mod, partner, etc
    user : str
        Username string including symbols mapping to badges

    Returns
    -------
    bool
    """
    return user_type in get_user_type_chatty(user)


def get_badges(line):
    """Get user badges

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        String with user badges
    """
    badges = ''
    if is_line_type("msg", line):
        for line in line.split(';'):
            if line.startswith('badges='):
                badges = re.search('badges=(.*)', line).group(1)
    return badges


def get_msg_id(line):
    """Get msg id

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        String with msg id
    """
    msg_id = ''
    for line in line.split(';'):
        if line.startswith('id='):
            msg_id = re.search('id=(.*)', line).group(1)
    return msg_id


def get_role(line):
    """Get user role from IRC line

    Parameters
    ----------
    line : str
        String including recent IRC message

    Returns
    -------
    str
        String with user role
    """
    if is_user_type_irc("mod", line):
        return "mod"
    if is_user_type_irc("vip", line):
        return "vip"
    if is_user_type_irc("sub", line):
        return "sub"
    if is_user_type_irc("pleb", line):
        return "pleb"
    if is_user_type_irc("partner", line):
        return "partner"


def is_user_type_irc(user_type, line):
    """Check if user has requested role

    Parameters
    ----------
    user_type : str
        User role to check. e.g. sub, mod, partner, etc
    line : str
        Line to check for requested role

    Returns
    -------
    bool
    """

    if not is_line_type("msg", line):
        return False

    badges = get_badges(line)
    user_type_map = {"sub": ["subscriber"],
                     "mod": ["moderator", "broadcaster"],
                     "partner": ["partner"],
                     "vip": ["vip"]}
    if user_type == "pleb":
        for _, v in user_type_map.items():
            if any(badge in badges for badge in v):
                return False
        return True
    else:
        return any(badge in badges for badge in user_type_map[user_type])


def get_message(line):
    """Get user message from line

    Parameters
    ----------
    line : str
        String including recent PubSub message

    Returns
    -------
    str
        user message
    """
    msg = ''
    for line in line.split(';'):
        if line.startswith('user-type='):
            msg = re.search('user-type=(.*)', line).group(1)
    return re.search(r'PRIVMSG #\w+ :(.*)', msg).group(1)
