"""IRC Socket Client"""
import asyncio
from datetime import datetime as dt
from datetime import timedelta

from modbot.utilities.logging import Logging, get_logger
from modbot.moderation import Moderation
from modbot.utilities.utilities import get_line_type
from modbot.connection.base import BaseSocketClientAsync, StreamHandlerAsync

logger = get_logger()


class IrcSocketClientAsync(Logging, Moderation, BaseSocketClientAsync):
    """Class to handle IRC connection"""
    _PING_MSG = "PING :tmi.twitch.tv"
    _PONG_OUT_MSG = "PONG :tmi.twitch.tv"
    _PONG_IN_MSG = "PONG tmi.twitch.tv"
    _HOST = 'irc.chat.twitch.tv'
    _PORT = 6667
    _WAIT_TIME = timedelta(seconds=300)
    VERBOSE_LOGGER = logger.irc_p
    EXTRA_VERBOSE_LOGGER = logger.irc_pp
    INFO_LOGGER = logger.irc

    def __init__(self, run_config):
        """
        Parameters
        ----------
        run_config : RunConfig
            Class with run time configuration parameters
        """
        Logging.__init__(self, run_config)
        Moderation.__init__(self, run_config)
        self.last_ping = dt.now()
        self.last_pong = dt.now()
        self.last_msg_time = dt.now()
        self.shandler = None
        self.run_config = run_config
        self.first_connection = True
        logger.update_level(run_config.LOGGER_LEVEL)
        self.INFO_LOGGER(f'{self.__name__} logger level: {logger.level}')

    @property
    def __name__(self):
        """Name of connection type"""
        return 'IRC'

    def _connect(self):
        """Send initial messages for IRC connection"""
        pwd = "PASS oauth:" + self.run_config._TOKEN
        self.shandler.write(pwd)
        nick = "NICK " + self.run_config.NICKNAME
        self.shandler.write(nick)
        chan = "JOIN #" + self.run_config.CHANNEL
        self.shandler.write(chan)
        line = "CAP REQ :twitch.tv/tags"
        self.shandler.write(line)
        line = "CAP REQ :twitch.tv/commands"
        self.shandler.write(line)
        line = "CAP REQ :twitch.tv/membership"
        self.shandler.write(line)

    def check_joins_and_parts(self, line):
        """Check user joins/parts to/from channel"""
        tmp = line.replace('tmi.twitch.tv', '').split(':')
        joined = [chunk.split('!')[0] for chunk in tmp if 'JOIN' in chunk]
        parted = [chunk.split('!')[0] for chunk in tmp if 'PART' in chunk]
        if joined:
            if self.first_connection:
                self.INFO_LOGGER(f"JOINED: {', '.join(joined)}")
                self.first_connection = False
            else:
                self.EXTRA_VERBOSE_LOGGER(f"JOINED: {', '.join(joined)}")
        if parted:
            self.EXTRA_VERBOSE_LOGGER(f"PARTED: {', '.join(parted)}")
        if not joined and not parted:
            self.EXTRA_VERBOSE_LOGGER(line)

    def handle_message(self, line):
        """Receive non chat IRC messages"""
        line_type = get_line_type(line)
        if self._PING_MSG in line:
            self.VERBOSE_LOGGER(f"IRC Ping: {dt.now()}")
            self.last_ping = dt.now()
            self.shandler.write(self._PONG_OUT_MSG)
            self.VERBOSE_LOGGER(f"IRC Pong: {dt.now()}")
            self.last_pong = dt.now()
        elif self._PONG_IN_MSG in line:
            self.VERBOSE_LOGGER(f"IRC Ping: {dt.now()}")
            self.last_ping = dt.now()
            self.VERBOSE_LOGGER(f"IRC Pong: {dt.now()}")
            self.last_pong = dt.now()
        elif line_type in ['join', 'part']:
            self.check_joins_and_parts(line)
        elif line_type in ['misc']:
            self.EXTRA_VERBOSE_LOGGER(line.strip('\n'))
        else:
            info = self.get_info_from_irc(line)
            if line_type in ['msg']:
                self.print_info(info)
            self._handle_message(info)

    def _update_user_log(self, info):
        """Update global chat history"""
        self.USER_LOG[info['user']] = self.USER_LOG.get(info['user'], [])
        self.USER_LOG[info['user']].append(dict(msg=info['msg'],
                                                prob=info['prob']))

    def _handle_message(self, info):
        """Handle chat IRC messages"""
        self.send_reply(self.shandler, info)
        self.send_action(self.shandler, info)
        log_entry = self.build_chat_log_entry(info)
        if info['deleted']:
            log_entry = self.build_action_log_entry(
                action='delete', user=info['user'],
                moderator=self.run_config.NICKNAME, msg=info['msg'],
                secs='', msg_id='')
            logger.mod(log_entry + f' ({info["prob"]})')

        self._update_user_log(info)
        # write to log
        try:
            self.append_log(log_entry)
        except Exception as e:
            msg = (f"**logging problem: {e}**")
            logger.warning(msg)

    def send_ping(self):
        """Send ping to keep connection alive"""
        self.shandler.write(self._PING_MSG)

    async def connect(self):
        """Initiate IRC connection"""
        self.INFO_LOGGER(f'**Trying to connect to {self.__name__}**')
        out = await asyncio.open_connection(self._HOST, self._PORT)
        self.shandler = StreamHandlerAsync(reader=out[0], writer=out[1])
        self._connect()
        loading = True
        while loading:
            line = await self.shandler.read(1024)
            loading = ("End of /NAMES list" in line)
            await asyncio.sleep(0.1)
        msg = f'**{self.__name__} connected to {self.run_config.CHANNEL}**'
        self.INFO_LOGGER(msg)

    async def receive_message(self):
        """Receive and handle IRC message"""
        now = dt.now()
        elapsed = now - self.last_msg_time
        if elapsed > self._WAIT_TIME:
            msg = f'{elapsed} since last message. Waiting on {self.__name__}.'
            self.EXTRA_VERBOSE_LOGGER(msg)
        try:
            line = await self.shandler.read(1024)
        except Exception as e:
            raise e
        self.last_msg_time = dt.now()
        self.handle_message(line)
        self.heartbeat()

    def quit(self):
        """Close stream handler connection"""
        self.shandler.close_connection()
