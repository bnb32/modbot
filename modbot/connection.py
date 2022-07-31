"""Module for handling IRC and PubSub connections"""
from abc import abstractmethod
import requests
from websockets.client import connect as ws_connect
from notify_run import Notify
import socket
import asyncio
import uuid
import json
from datetime import datetime as dt
from datetime import timedelta

from modbot.utilities.logging import Logging, get_logger
from modbot.moderation import Moderation
from modbot.utilities.utilities import get_line_type

logger = get_logger()


class StreamHandler:
    """Class to handle reading from and writing to stream"""
    __HOST = 'irc.chat.twitch.tv'
    __PORT = 6667
    __SOCKET = None

    def __init__(self, writer=None, reader=None):
        if writer is None or reader is None:
            self.__SOCKET = socket.socket()
            self.__SOCKET.connect((self.__HOST, self.__PORT))
            self._write = self.__SOCKET.send
            self._read = self.__SOCKET.recv
            logger.info(f'Connected to {self.__HOST} on port {self.__PORT}')
        else:
            self._write = writer.write
            self._read = reader.read

    def write(self, message):
        """Send IRC message"""
        msg = message + '\r\n'
        msg = msg.encode('utf-8')
        self._write(msg)

    def read(self, buffer_size):
        """Read IRC message"""
        data = b''
        while True:
            part = self._read(buffer_size)
            data += part
            if len(part) < buffer_size:
                break
        data = data.decode('utf-8')
        return data

    def close_connection(self):
        """Close the connection"""
        if self.__SOCKET is not None:
            self.__SOCKET.close()
            logger.info('IRC connection closed')


class StreamHandlerAsync(StreamHandler):
    """A stream handler class for asynchronous IO."""

    async def read(self, buffer_size):
        """Read IRC message"""
        data = b''
        while True:
            part = await self._read(buffer_size)
            data += part
            if len(part) < buffer_size:
                break
        data = data.decode('utf-8')
        return data


class IrcSocketClientMixIn(Logging, Moderation):
    """Class with base methods for IrcSocketClient subclasses"""
    _PING_MSG = "PING :tmi.twitch.tv"
    _PONG_OUT_MSG = "PONG :tmi.twitch.tv"
    _PONG_IN_MSG = "PONG tmi.twitch.tv"
    _HOST = 'irc.chat.twitch.tv'
    _PORT = 6667
    _WAIT_TIME = timedelta(seconds=300)

    def __init__(self, run_config):
        """
        Parameters
        ----------
        run_config : RunConfig
            Class with run time configuration parameters
        """
        Logging.__init__(self, run_config)
        Moderation.__init__(self, run_config)
        self.notify = Notify()
        self.last_ping = dt.now()
        self.last_pong = dt.now()
        self.last_msg_time = dt.now()
        self.shandler = None
        self.run_config = run_config

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

    def handle_message(self, line):
        """Receive non chat IRC messages"""
        if self._PING_MSG in line:
            logger.verbose(f"IRC Ping: {dt.now()}")
            self.last_ping = dt.now()
            self.shandler.write(self._PONG_OUT_MSG)
            logger.verbose(f"IRC Pong: {dt.now()}")
            self.last_pong = dt.now()
        elif self._PONG_IN_MSG in line:
            logger.verbose(f"IRC Ping: {dt.now()}")
            self.last_ping = dt.now()
            logger.verbose(f"IRC Pong: {dt.now()}")
            self.last_pong = dt.now()
        elif get_line_type(line) in ['join', 'part']:
            tmp = line.replace('tmi.twitch.tv', '').split(':')
            joined = [chunk.split('!')[0] for chunk in tmp if 'JOIN' in chunk]
            parted = [chunk.split('!')[0] for chunk in tmp if 'PART' in chunk]
            if joined:
                logger.extra_verbose(f"JOINED: {', '.join(joined)}")
            if parted:
                logger.extra_verbose(f"PARTED: {', '.join(parted)}")
            if not joined and not parted:
                logger.extra_verbose(line)
        elif get_line_type(line) not in ['ban', 'delete', 'msg']:
            logger.extra_verbose(line)
        else:
            self._handle_message(line)

    def heartbeat(self):
        """Heartbeat routine for keeping IRC connection alive"""
        if dt.now() - self.last_ping > self._WAIT_TIME:
            logger.verbose(f"IRC Ping: {dt.now()}")
            self.last_ping = dt.now()
            self.shandler.write(self._PING_MSG)
            logger.verbose(f"IRC Pong: {dt.now()}")
            self.last_pong = dt.now()
        else:
            pass

    def _handle_message(self, line):
        """Handle chat IRC messages"""
        info = self.get_info_from_irc(line)
        self.print_info(info)
        self.send_reply(self.shandler, info)
        self.send_action(self.shandler, info)
        log_entry = self.build_chat_log_entry(info)
        if info['deleted']:
            log_entry = self.build_action_log_entry('delete', info['user'],
                                                    self.run_config.NICKNAME,
                                                    info['msg'], '', '')
            logger.mod(log_entry + f' ({info["prob"]})')
        self.UserLog[info['user']] = info['msg']
        # write to log
        try:
            self.append_log(log_entry)
        except Exception:
            logger.warning("**logging problem**")
            logger.warning(line)


class BaseSocketClientAsync:

    @property
    def __name__(self):
        return 'BaseSocketClientAsync'

    @abstractmethod
    def receive_message(self):
        """Receive and handle socket message"""

    @abstractmethod
    def connect(self):
        """Connection to socket"""

    @abstractmethod
    def heartbeat(self):
        """Maintain socket connection"""

    def connect_fail(self, e):
        """Response to connection failure"""
        msg = f'**{self.__name__} Ping failed: {e}**'
        logger.verbose(msg)

    def receive_fail(self, e):
        """Response to message receive failure"""
        msg = (f'Exception while receiving {self.__name__} message: {e}')
        logger.extra_verbose(msg)

    async def listen_forever(self):
        """Listen for socket connection"""
        while True:
            await self.connect()
            try:
                while True:
                    try:
                        await self.receive_message()
                    except Exception as e:
                        self.receive_fail(e)
                        try:
                            await self.heartbeat()
                            continue
                        except Exception as e:
                            self.connect_fail(e)
                            break

            except KeyboardInterrupt as e:
                logger.warning(f'Received exit, exiting {self.__name__}: {e}')

            except Exception as e:
                msg = f'Unknown problem with {self.__name__} connection: {e}'
                logger.warning(msg)
                raise RuntimeError(msg) from e


class IrcSocketClientAsync(IrcSocketClientMixIn, BaseSocketClientAsync):
    """Class to handle IRC connection"""

    async def connect(self):
        """Initiate IRC connection"""
        logger.info(f'**Trying to connect to {self.__name__}**')
        out = await asyncio.open_connection(self._HOST, self._PORT)
        self.shandler = StreamHandlerAsync(reader=out[0], writer=out[1])
        self._connect()
        loading = True
        while loading:
            line = await self.shandler.read(1024)
            loading = ("End of /NAMES list" in line)
            await asyncio.sleep(0.1)
        msg = f'**{self.__name__} connected to {self.run_config.CHANNEL}**'
        logger.info(msg)

    async def _heartbeat(self):
        """Keep IRC connection alive"""
        super().heartbeat()

    async def receive_message(self):
        """Receive and handle IRC message"""
        now = dt.now()
        elapsed = now - self.last_msg_time
        if elapsed > self._WAIT_TIME:
            msg = f'{elapsed} since last message. Waiting on {self.__name__}.'
            logger.extra_verbose(msg)
        line = await self.shandler.read(1024)
        self.last_msg_time = dt.now()
        self.handle_message(line)

    def connect_fail(self, e):
        """Response to connection failure"""
        logger.info(f'**{self.__name__} Ping failed: {e}**')
        self.notify.send(f"{self.__name__} disconnnected.")


class WebSocketClientAsync(Logging, BaseSocketClientAsync):
    """Class to handle PubSub connection"""

    _AUTH_URL = "https://id.twitch.tv/oauth2/token"
    _USER_URL = "https://api.twitch.tv/helix/users?login={user}"
    _URI = 'wss://pubsub-edge.twitch.tv'
    _PING_TIMEOUT = timedelta(seconds=60)
    _WAIT_TIME = timedelta(seconds=300)
    _PRINT_TIME = timedelta(seconds=600)

    def __init__(self, run_config):
        """
        Parameters
        ----------
        run_config : RunConfig
            Class with run time configuration parameters
        """
        super().__init__(run_config)
        self.run_config = run_config
        self.topics = ["chat_moderator_actions.{}.{}"
                       .format(self.moderator_id, self.channel_id)]
        self.auth_token = self.run_config._TOKEN
        self.message = {}
        self.last_ping = dt.now()
        self.last_pong = dt.now()
        self.last_print = dt.now()
        self.last_msg_time = dt.now()
        self.connected = False
        self.connection = None

    @property
    def __name__(self):
        """Name of connection type"""
        return 'PubSub'

    def get_user_id(self, user):
        """Get moderator user id

        Parameters
        ----------
        user : str
            User for which to get id

        Returns
        -------
        str
            User id
        """
        AutParams = {'client_id': self.run_config._CLIENT_ID,
                     'client_secret': self.run_config._CLIENT_SECRET,
                     'grant_type': 'client_credentials'}
        AutCall = requests.post(url=self._AUTH_URL, params=AutParams)
        access_token = AutCall.json()['access_token']
        head = {'Client-ID': self.run_config._CLIENT_ID,
                'Authorization': "Bearer " + access_token}
        r = requests.get(self._USER_URL.format(user=user),
                         headers=head).json()['data']
        return r[0]['id']

    @property
    def channel_id(self):
        """ID for channel to moderate"""
        return self.get_user_id(self.run_config.CHANNEL)

    @property
    def moderator_id(self):
        """ID for bot username"""
        return self.get_user_id(self.run_config.NICKNAME)

    async def heartbeat(self):
        """Keep PubSub connection alive"""
        self.last_ping = dt.now()
        pong = await self.connection.ping()
        await asyncio.wait_for(pong, timeout=self._PING_TIMEOUT.seconds)
        self.last_pong = dt.now()
        if dt.now() - self.last_print > self._PRINT_TIME:
            self.last_print = dt.now()
            logger.verbose(f'{self.__name__} Ping: {self.last_ping}')
            logger.verbose(f'{self.__name__} Pong: {self.last_pong}')

    async def connect(self):
        """Report initial connection"""
        logger.info(f'**Trying to connect to {self.__name__}**')
        self.connection = await ws_connect(self._URI)
        if self.connection.open:
            msg = f'**{self.__name__} connected to {self.run_config.CHANNEL}**'
            logger.info(msg)
            if not self.connected:
                self.connected = True
            message = {"type": "LISTEN",
                       "nonce": str(self.generate_nonce()),
                       "data": {"topics": self.topics,
                                "auth_token": self.auth_token}}
            json_message = json.dumps(message)
            await self.connection.send(json_message)

    async def receive_message(self):
        """Recieve PubSub message"""
        elapsed = dt.now() - self.last_msg_time
        if elapsed > self._WAIT_TIME:
            msg = f'{elapsed} since last message. Waiting on {self.__name__}.'
            logger.extra_verbose(msg)
        message = await asyncio.wait_for(self.connection.recv(),
                                         timeout=self._WAIT_TIME.seconds)
        self.last_msg_time = dt.now()
        msg = f'{self.__name__} message received: {self.last_msg_time}'
        logger.verbose(msg)
        self.message = message
        await self.handle_message(message)

    async def handle_message(self, message):
        """Handle moderation decisions based on PubSub message

        Parameters
        ----------
        message : str
            String containing PubSub message
        """
        tmp = json.loads(message)
        if 'PONG' in tmp['type']:
            self.last_pong = dt.now()
        if 'MESSAGE' in tmp['type']:
            tmp = json.loads(tmp['data']['message'])
            tmp = tmp['data']
            out = self.get_info_from_pubsub(tmp)
            action, user, moderator, msg, secs, msg_id = out
            log_entry = self.build_action_log_entry(action, user, moderator,
                                                    msg, secs, msg_id)
            logger.mod(log_entry)
            try:
                self.append_log(log_entry)
            except Exception:
                logger.warning("**logging problem**")
                logger.warning(log_entry)

    @staticmethod
    def generate_nonce():
        """Generate nonce value. Needed for PubSub connection."""
        nonce = uuid.uuid1()
        oauth_nonce = nonce.hex
        return oauth_nonce
