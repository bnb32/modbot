"""PubSub Socket Client"""

import requests
from websockets import connect as ws_connect
import uuid
import json
from datetime import datetime as dt
from datetime import timedelta
import asyncio

from modbot.utilities.logging import Logging, get_logger
from modbot.connection.base import BaseSocketClientAsync

logger = get_logger()


class WebSocketClientAsync(Logging, BaseSocketClientAsync):
    """Class to handle PubSub connection"""

    _AUTH_URL = "https://id.twitch.tv/oauth2/token"
    _USER_URL = "https://api.twitch.tv/helix/users?login={user}"
    _URI = 'wss://pubsub-edge.twitch.tv'
    _PING_TIMEOUT = timedelta(seconds=60)
    _WAIT_TIME = timedelta(seconds=300)
    _PING_MSG = json.dumps({'TYPE': 'PING'})
    VERBOSE_LOGGER = logger.pubsub_p
    EXTRA_VERBOSE_LOGGER = logger.pubsub_pp
    INFO_LOGGER = logger.pubsub

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
        self.last_ping = dt.now()
        self.last_pong = dt.now()
        self.last_msg_time = dt.now()
        self.connected = False
        self.connection = None
        logger.update_level(run_config.LOGGER_LEVEL)
        self.INFO_LOGGER(f'{self.__name__} logger level: {logger.level}')

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

    async def send_ping(self):
        """Send ping to keep connection alive"""
        await self.connection.send(self._PING_MSG)

    async def heartbeat(self):
        """Keep PubSub connection alive"""
        self.VERBOSE_LOGGER(f"{self.__name__} Ping: {dt.now()}")
        self.last_ping = dt.now()
        await self.send_ping()
        await asyncio.sleep(self._WAIT_TIME.seconds)

    async def connect(self):
        """Report initial connection"""
        self.INFO_LOGGER(f'**Trying to connect to {self.__name__}**')
        self.connection = await ws_connect(self._URI, ping_interval=None)
        if self.connection.open:
            msg = f'**{self.__name__} connected to {self.run_config.CHANNEL}**'
            self.INFO_LOGGER(msg)
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
        msg = f'{elapsed} since last message. Waiting on {self.__name__}.'
        self.EXTRA_VERBOSE_LOGGER(msg)
        message = await self.connection.recv()
        self.last_msg_time = dt.now()
        msg = f'{self.__name__} message received: {self.last_msg_time}'
        self.VERBOSE_LOGGER(msg)
        await self.handle_message(message)
        await self.heartbeat()

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
            self.VERBOSE_LOGGER(f"{self.__name__} Pong: {dt.now()}")
            self.last_pong = dt.now()
        elif 'MESSAGE' in tmp['type']:
            tmp = json.loads(tmp['data']['message'])
            tmp = tmp['data']
            out = self.get_info_from_pubsub(tmp)
            log_entry = self.build_action_log_entry(*out[:-1])
            logger.mod(log_entry.replace('\n', ' ') + f' ({out[-1]})')
            try:
                self.append_log(log_entry)
            except Exception as e:
                logger.warning(f"**logging problem: {e}**")
                logger.warning(log_entry)
        else:
            msg = f'Received {self.__name__} message: {tmp}'
            self.EXTRA_VERBOSE_LOGGER(msg)

    @staticmethod
    def generate_nonce():
        """Generate nonce value. Needed for PubSub connection."""
        nonce = uuid.uuid1()
        oauth_nonce = nonce.hex
        return oauth_nonce

    def quit(self):
        """Close pubsub connection"""
        self.connection.close()
