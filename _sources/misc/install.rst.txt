Clone repo (recommended for developers)
---------------------------------------

1. from home dir, ``git clone git@github.com:bnb32/modbot.git``

2. Create ``modbot`` environment and install package
    1) Create a conda env: ``conda create -n modbot``
    2) Run the command: ``conda activate modbot``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``modbot`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)
    6) To train on a GPU run ``pip install tensorflow-gpu`` and
       ``conda install -c conda-forge cudnn``
    7) If using a BERT model download a preprocessing model and encoder and put
       in ``data/bert_preprocess`` and ``data/bert_encoder``, respectively.
       Example downloads: ``https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3``
       and ``https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4``