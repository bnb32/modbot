"""Module for handling IRC and PubSub connections"""
import requests
from websockets.client import connect as ws_connect
from notify_run import Notify
import socket
import asyncio
import uuid
import json
import time
import sys

from modbot.utilities.logging import Logging, get_logger
from modbot.moderation import Moderation
from modbot.utilities.utilities import get_line_type, date_time

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
    _WAIT_TIME = 300

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
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.last_msg_time = time.time()
        self.shandler = None
        self.run_config = run_config

    def _connect(self):
        """Send initial messages for IRC connection"""
        logger.info('**Trying to establish IRC connection to '
                    f'{self.run_config.CHANNEL}**')
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
            logger.verbose(f"**IRC Ping: {date_time()}**")
            self.last_ping = time.time()
            self.shandler.write(self._PONG_OUT_MSG)
            logger.verbose(f"**IRC Pong: {date_time()}**")
            self.last_pong = time.time()
            logger.verbose('**IRC connection still alive**')
        elif self._PONG_IN_MSG in line:
            logger.verbose(f"**IRC Ping: {date_time()}**")
            self.last_ping = time.time()
            logger.verbose(f"**IRC Pong: {date_time()}**")
            self.last_pong = time.time()
            logger.verbose('**IRC connection still alive**')
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
        if time.time() >= (self.last_ping + self._WAIT_TIME):
            logger.verbose(f"**IRC Ping: {date_time()}**")
            self.last_ping = time.time()
            self.shandler.write(self._PING_MSG)
            logger.verbose(f"**IRC Pong: {date_time()}**")
            self.last_pong = time.time()
            logger.verbose('**IRC Ping OK, keeping connection alive...**')
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


class IrcSocketClient(IrcSocketClientMixIn):
    """Class to handle IRC connection"""

    def quit(self):
        """Close IRC socket client"""
        self.shandler.close_connection()

    def connect(self):
        """Initiate IRC connection"""
        self._connect()
        loading = True
        while loading:
            line = self.shandler.read(1024)
            loading = ("End of /NAMES list" in line)

        logger.info('**IRC-Client correctly connected to '
                    f'{self.run_config.CHANNEL}**')

    def listen_forever(self):
        """Listen for IRC connection"""
        self.connect()
        try:
            while True:
                try:
                    logger.extra_verbose('**Waiting for IRC message**')
                    line = self.shandler.read(1024)
                    self.handle_message(line)
                except socket.timeout:
                    try:
                        self.heartbeat()
                        continue
                    except Exception:
                        logger.info('**IRC Ping failed**')
                        self.notify.send("Chatbot disconnnected")
                        break

        except KeyboardInterrupt:
            logger.warning('Received exit, exiting IRC')

        except Exception as e:
            logger.warning(f'Unknown problem with IRC connection: {e}')
            raise e


class IrcSocketClientAsync(IrcSocketClientMixIn):
    """Class to handle IRC connection"""

    async def connect(self):
        """Initiate IRC connection"""
        logger.info('**Trying to connect to IRC**')
        out = await asyncio.open_connection(self._HOST, self._PORT)
        self.shandler = StreamHandlerAsync(reader=out[0], writer=out[1])
        self._connect()
        loading = True
        while loading:
            line = await self.shandler.read(1024)
            loading = ("End of /NAMES list" in line)
            await asyncio.sleep(0.1)
        logger.info('**IRC Connection established. IRC-Client correctly '
                    f'connected to {self.run_config.CHANNEL}**')

    async def _heartbeat(self):
        """Keep IRC connection alive"""
        super().heartbeat()

    async def listen_forever(self):
        """Listen for IRC connection"""
        await self.connect()
        try:
            while True:
                try:
                    logger.extra_verbose('**Waiting for IRC message**')
                    line = await self.shandler.read(1024)
                    self.handle_message(line)
                except Exception:
                    try:
                        self.heartbeat()
                        continue
                    except Exception:
                        logger.info('**IRC Ping failed**')
                        self.notify.send("Chatbot disconnnected")
                        break

        except KeyboardInterrupt:
            logger.warning('Received exit, exiting IRC')

        except Exception as e:
            logger.warning(f'Unknown problem with IRC connection: {e}')
            raise e


class WebSocketClientAsync(Logging):
    """Class to handle PubSub connection"""

    _AUTH_URL = "https://id.twitch.tv/oauth2/token"
    _USER_URL = "https://api.twitch.tv/helix/users?login={user}"
    _URI = 'wss://pubsub-edge.twitch.tv'
    _PING_TIMEOUT = 60
    _WAIT_TIME = 300
    _PRINT_TIME = 600

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
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.last_print = time.time()
        self.last_msg_time = time.time()
        self.connected = False
        self.connection = None

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
        self.last_ping = date_time()
        pong = await self.connection.ping()
        await asyncio.wait_for(pong, timeout=self._PING_TIMEOUT)
        self.last_pong = date_time()
        if time.time() > (self.last_print + self._PRINT_TIME):
            self.last_print = date_time()
            logger.verbose('**PubSub Ping: %s**' % self.last_ping)
            logger.verbose('**PubSub Pong: %s**' % self.last_pong)
            logger.verbose('**PubSub Ping OK, keeping connection alive...**')

    def connection_status(self):
        """Report connection status"""
        if not self.connected:
            logger.info('**Trying to connect to PubSub**')
        else:
            logger.verbose('**Connected to PubSub**')

    async def connect(self):
        """Report initial connection"""
        if not self.connected:
            logger.info('**PubSub Connection established. '
                        'Web-Client correctly connected**')
            self.connected = True
        else:
            logger.verbose('**PubSub Connection established. '
                           'Web-Client correctly connected**')
        message = {"type": "LISTEN",
                   "nonce": str(self.generate_nonce()),
                   "data": {"topics": self.topics,
                            "auth_token": self.auth_token}}
        json_message = json.dumps(message)
        await self.connection.send(json_message)

    async def receive_message(self):
        """Recieve PubSub message"""
        logger.verbose('**Waiting for PubSub message**')
        message = await asyncio.wait_for(self.connection.recv(),
                                         timeout=self._WAIT_TIME)
        self.last_msg_time = date_time()
        logger.verbose('**Message received: %s**' % self.last_msg_time)

        self.message = message
        await self.handle_message(message)

    async def listen_forever(self):
        """Listen for PubSub connection"""
        while True:
            try:
                logger.info('**Trying to connect to PubSub**')
                async with ws_connect(self._URI) as self.connection:
                    if self.connection.open:
                        await self.connect()
                        while True:
                            try:
                                await self.receive_message()
                            except Exception:
                                try:
                                    await self.heartbeat()
                                    continue
                                except Exception:
                                    logger.verbose('**PubSub Ping failed**')
                                    break  # inner loop

            except Exception:
                logger.warning('Problem with PubSub connection')
                sys.exit()

    async def handle_message(self, message):
        """Handle moderation decisions based on PubSub message

        Parameters
        ----------
        message : str
            String containing PubSub message
        """
        tmp = json.loads(message)

        if 'PONG' in tmp['type']:
            self.last_pong = time.time()

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
