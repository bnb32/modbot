"""Logging tests"""
from emoji import demojize
import json

from modbot.utilities.logging import Logging

MSG_DICT = '{"type": "MESSAGE", "data": {"topic": '
MSG_DICT += '"chat_moderator_actions.420749209.127550308", '
MSG_DICT += '"message": {"type":"moderation_action",'
MSG_DICT += '"data":{"type":"chat_login_moderation",'
MSG_DICT += '"moderation_action":"delete",'
MSG_DICT += '"args":["nowissno","i usually play chess while taking a massive '
MSG_DICT += 'shit", "374534ad-a3b5-426b-be65-12cf869d5241"],'
MSG_DICT += '"created_by":"gerberbaby4", "created_by_user_id":"711705717",'
MSG_DICT += '"created_at":"2022-05-27T23:18:06.964295963Z",'
MSG_DICT += '"msg_id":"", "target_user_id":"532364796","target_user_login":"",'
MSG_DICT += '"from_automod":false}}}}'


def test_get_info_from_msg():
    """Test PubSub message parsing"""
    msg_dict = json.loads(MSG_DICT)
    msg_dict = msg_dict['data']['message']
    msg_dict = msg_dict['data']
    action = Logging.get_value('moderation_action', msg_dict)
    moderator = Logging.get_value('created_by', msg_dict)

    if 'args' in msg_dict:
        user = msg_dict['args'][0]
    else:
        user = Logging.get_value('target_user_login', msg_dict)

    if "delete" in action:
        msg = demojize(msg_dict['args'][1])
        msg_id = msg_dict['args'][2]

    assert user == 'nowissno'
    assert moderator == 'gerberbaby4'
    assert msg == 'i usually play chess while taking a massive shit'
    assert msg_id == '374534ad-a3b5-426b-be65-12cf869d5241'
