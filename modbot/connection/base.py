"""Module for handling IRC and PubSub connections"""
from abc import abstractmethod
import socket
from datetime import datetime as dt

from modbot.utilities.logging import get_logger

logger = get_logger()


class StreamHandlerAsync:
    """A stream handler class for asynchronous IO."""

    """Class to handle reading from and writing to stream"""
    __HOST = 'irc.chat.twitch.tv'
    __PORT = 6667
    __SOCKET = None
    INFO_LOGGER = logger.irc

    def __init__(self, writer=None, reader=None):
        if writer is None or reader is None:
            self.__SOCKET = socket.socket()
            self.__SOCKET.connect((self.__HOST, self.__PORT))
            self._write = self.__SOCKET.send
            self._read = self.__SOCKET.recv
            self.INFO_LOGGER(f'Connected to {self.__HOST} on port '
                             f'{self.__PORT}')
        else:
            self.writer = writer
            self.reader = reader
            self._write = writer.write
            self._read = reader.read

    def write(self, message):
        """Send IRC message"""
        msg = message + '\r\n'
        msg = msg.encode('utf-8')
        self._write(msg)

    def close_connection(self):
        """Close the connection"""
        if self.__SOCKET is not None:
            self.__SOCKET.close()
            self.INFO_LOGGER('IRC connection closed')
        else:
            self.writer.close()

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


class BaseSocketClientAsync:

    VERBOSE_LOGGER = logger.verbose
    EXTRA_VERBOSE_LOGGER = logger.extra_verbose
    INFO_LOGGER = logger.info
    _PING_MSG = None

    @property
    def __name__(self):
        return 'BaseSocketClientAsync'

    @abstractmethod
    def send_ping(self):
        """Send ping to keep connection alive"""

    @abstractmethod
    def receive_message(self):
        """Receive and handle socket message"""

    @abstractmethod
    def connect(self):
        """Connection to socket"""

    def heartbeat(self):
        """Heartbeat routine for keeping connection alive"""
        if dt.now() - self.last_ping > self._WAIT_TIME:
            self.VERBOSE_LOGGER(f"{self.__name__} Ping: {dt.now()}")
            self.last_ping = dt.now()
            self.send_ping()
            self.VERBOSE_LOGGER(f"{self.__name__} Pong: {dt.now()}")
            self.last_pong = dt.now()
        else:
            pass

    def connect_fail(self, e):
        """Response to connection failure"""
        msg = f'**{self.__name__} Ping failed: {e}**'
        self.VERBOSE_LOGGER(msg)

    def receive_fail(self, e):
        """Response to message receive failure"""
        msg = (f'Exception while receiving {self.__name__} message: {e}')
        self.EXTRA_VERBOSE_LOGGER(msg)

    async def listen_forever(self):
        """Listen for socket connection"""
        while True:
            await self.connect()
            try:
                while True:
                    try:
                        await self.receive_message()
                    except Exception as e:
                        self.connect_fail(e)
                        break

            except KeyboardInterrupt as e:
                logger.warning(f'Received exit, exiting {self.__name__}: {e}')

            except Exception as e:
                msg = f'Unknown problem with {self.__name__} connection: {e}'
                logger.warning(msg)
                raise RuntimeError(msg) from e
