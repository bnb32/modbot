"""Module for handling IRC and PubSub connections"""
import requests
from websockets.client import connect as ws_connect
from notify_run import Notify
import asyncio
import uuid
import json
import time
import sys

from modbot.utilities.logging import Logging, get_logger
from modbot.moderation import Moderation
from modbot.utilities.utilities import get_line_type, date_time

logger = get_logger()

ping_msg = "PING :tmi.twitch.tv\r\n"
pong_out_msg = "PONG :tmi.twitch.tv\r\n"
pong_in_msg = "PONG tmi.twitch.tv"

authURL = "https://id.twitch.tv/oauth2/token"
userURL = "https://api.twitch.tv/helix/users?login=%s"


class IrcSocketClient(Logging, Moderation):
    """Class to handle IRC connection"""
    def __init__(self, run_config):
        """
        Parameters
        ----------
        run_config : RunConfig
            Class with run time configuration parameters
        """
        Logging.__init__(self, run_config)
        Moderation.__init__(self, run_config)
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.last_msg_time = time.time()
        self.ping_timeout = 60
        self.sleep_time = 60
        self.wait_time = 300
        self.uri = 'irc.chat.twitch.tv'
        self.port = 6667
        self.reader = None
        self.writer = None
        self.notify = Notify()
        self.run_config = run_config

    async def connect(self):
        """Initiate IRC connection"""
        try:
            self.writer.write(ping_msg.encode("utf-8"))
        except Exception:
            logger.info('**Trying to connect to IRC**')
            self.reader, self.writer = await asyncio.open_connection(self.uri,
                                                                     self.port)
            pwd = "PASS oauth:" + self.run_config._TOKEN + "\r\n"
            self.writer.write(pwd.encode("utf-8"))
            nick = "NICK " + self.run_config.NICKNAME + "\r\n"
            self.writer.write(nick.encode("utf-8"))
            chan = "JOIN #" + self.run_config.CHANNEL + "\r\n"
            self.writer.write(chan.encode("utf-8"))

            loading = True
            line = "CAP REQ :twitch.tv/tags\r\n"
            self.writer.write(line.encode("utf-8"))
            line = "CAP REQ :twitch.tv/commands\r\n"
            self.writer.write(line.encode("utf-8"))
            line = "CAP REQ :twitch.tv/membership\r\n"
            self.writer.write(line.encode("utf-8"))

            while loading:
                line = await self.reader.read(1024)
                line = line.decode("utf-8")
                loading = ("End of /NAMES list" in line)
                await asyncio.sleep(0.1)

            logger.info('**IRC Connection established. IRC-Client correctly '
                        f'connected to {self.run_config.CHANNEL}**')

    async def receive_message(self):
        """Receive IRC message"""
        logger.extra_verbose('**Waiting for IRC message**')
        line = await self.reader.read(1024)
        line = line.decode("utf-8")

        # keep connection
        if line == ping_msg:
            logger.verbose(f"**IRC Ping: {date_time()}**")
            self.last_ping = time.time()
            self.writer.write(pong_out_msg.encode("utf-8"))
            logger.verbose(f"**IRC Pong: {date_time()}**")
            self.last_pong = time.time()
            logger.verbose('**IRC connection still alive**')
        elif pong_in_msg in line:
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
            # get info on line
            await self.handle_message(line)

    async def heartbeat(self):
        """Keep IRC connection alive"""
        if time.time() >= (self.last_ping + self.wait_time):
            logger.verbose(f"**IRC Ping: {date_time()}**")
            self.last_ping = time.time()
            self.writer.write(ping_msg.encode("utf-8"))
            logger.verbose(f"**IRC Pong: {date_time()}**")
            self.last_pong = time.time()
            logger.verbose('**IRC Ping OK, keeping connection alive...**')
        else:
            pass

    async def listen_forever(self):
        """Listen for IRC connection"""
        while True:
            try:
                await self.connect()
                while True:
                    try:
                        await self.receive_message()
                    except Exception:
                        try:
                            await self.heartbeat()
                            continue
                        except Exception:
                            logger.info('**IRC Ping failed**')
                            self.notify.send("Chatbot disconnnected")
                            break

            except KeyboardInterrupt:
                logger.warning('Received exit, exiting IRC')
                sys.exit()

            except Exception:
                logger.warning('Problem with IRC connection')
                sys.exit()

    async def handle_message(self, line):
        """Handle moderation decisions based on IRC message"""
        info = self.get_info_from_irc(line)
        self.print_info(info)
        self.send_reply(self.writer, info)
        self.send_action(self.writer, info)

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


class WebSocketClient(Logging):
    """Class to handle PubSub connection"""
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
        self.last_msg_time = time.time()
        self.ping_timeout = 60
        self.sleep_time = 60
        self.wait_time = 300
        self.uri = 'wss://pubsub-edge.twitch.tv'
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
        AutCall = requests.post(url=authURL, params=AutParams)
        access_token = AutCall.json()['access_token']
        head = {'Client-ID': self.run_config._CLIENT_ID,
                'Authorization': "Bearer " + access_token}
        r = requests.get(userURL % user, headers=head).json()['data']
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
        logger.verbose('**PubSub Ping: %s**' % self.last_ping)
        pong = await self.connection.ping()
        await asyncio.wait_for(pong, timeout=self.ping_timeout)
        self.last_pong = date_time()
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
                                         timeout=self.wait_time)
        self.last_msg_time = date_time()
        logger.verbose('**Message received: %s**' % self.last_msg_time)

        self.message = message
        await self.handle_message(message)

    async def listen_forever(self):
        """Listen for PubSub connection"""
        while True:
            try:
                logger.info('**Trying to connect to PubSub**')
                async with ws_connect(self.uri) as self.connection:
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
