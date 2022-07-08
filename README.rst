*********************
Welcome to Modbot!
*********************

.. image:: https://github.com/bnb32/modbot/workflows/Documentation/badge.svg
    :target: https://bnb32.github.io/modbot/

.. image:: https://github.com/bnb32/modbot/workflows/Pytests/badge.svg
    :target: https://github.com/bnb32/modbot/actions?query=workflow%3A%22Pytests%22

.. image:: https://github.com/bnb32/modbot/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/bnb32/modbot/actions?query=workflow%3A%22Lint+Code+Base%22

.. image:: https://codecov.io/gh/bnb32/modbot/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/bnb32/modbot


Installation
============

Follow instructions `here <https://bnb32.github.io/modbot/misc/install.html>`_


Environment variables
=====================

Register bot with twitch and get Client ID and Client Secret `here <https://dev.twitch.tv/console/apps>`_.


Get `Oauth Token <https://twitchapps.com/tmi/>`_.


Update variables in ``config.json`` and ``__init__.py``.

.. code-block:: bash

    cd modbot/environment/
    cp config.json ../../my_config.json
    vim my_config.json
    vim __init__.py
    cd ../../

Training Model
==============

From scratch with classified messages in csv file
(with columns ``text``, ``is_offensive``):

.. code-block:: bash

    python -m modbot.training -train -infile <messages.csv> -config my_config.json

Train from scratch from chatty data:

.. code-block:: bash

    python -m modbot.training -infile <infile> -train -clean -config my_config.json

Retrain with additional chatty data:

.. code-block:: bash

    python -m modbot.training -append -infile <infile> -train -clean -config my_config.json


Running
=======

Run bot:

.. code-block:: bash

    python -m modbot -config my_config.json


.. inclusion-intro
