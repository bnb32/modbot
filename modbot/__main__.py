"""Main entry for running bot"""
import sys
import pprint
import asyncio

from modbot.connection import WebSocketClient, IrcSocketClient
from modbot.utilities.logging import get_logger
from modbot import modbot_argparse
from modbot.environment import RunConfig


if __name__ == '__main__':
    parser = modbot_argparse()
    args = parser.parse_args()
    config = RunConfig(args=args)
    logger = get_logger(config.LOGGER_LEVEL)

    ircClient = IrcSocketClient(config)
    webClient = WebSocketClient(config)

    msg = ('Running with configuration:\n'
           f'{pprint.pformat(config.upper_attrs, indent=1)}')
    logger.info(msg)

    try:
        loop = asyncio.new_event_loop()
        loop = asyncio.get_event_loop()
        loop.create_task(ircClient.listen_forever())
        loop.create_task(webClient.listen_forever())
        loop.run_forever()

    except KeyboardInterrupt:
        logger.warning("Exiting modbot!")
        sys.exit()
