"""Main entry for running bot"""
import sys
import pprint
import asyncio

from modbot.connection import WebSocketClientAsync, IrcSocketClientAsync
from modbot.utilities.logging import get_logger
from modbot import modbot_argparse
from modbot.environment import RunConfig


def main():
    parser = modbot_argparse()
    args = parser.parse_args()
    config = RunConfig(args=args)
    logger = get_logger()
    logger.update_level(config.LOGGER_LEVEL)
    logger.info(f'Initialized main logger with level: {logger.level}')

    ircClient = IrcSocketClientAsync(config)
    webClient = WebSocketClientAsync(config)

    msg = ('Running with configuration:\n'
           f'{pprint.pformat(config.public_attrs, indent=1)}')
    logger.info(msg)
    try:
        loop = asyncio.new_event_loop()
        loop = asyncio.get_event_loop()
        loop.create_task(webClient.listen_forever())
        loop.create_task(ircClient.listen_forever())
        loop.run_forever()
    except KeyboardInterrupt:
        logger.warning("Exiting modbot!")
        ircClient.quit()
        sys.exit()


if __name__ == '__main__':
    main()
