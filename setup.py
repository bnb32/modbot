"""Setup script"""
from distutils.core import setup

setup(name='modbot', version='0.1.0',
      url='https://github.com/bnb32/modbot',
      author='Brandon N. Benton',
      description='twitch moderation bot',
      packages=['modbot'],
      package_dir={'modbot': './modbot'},
      entry_points={
          "console_scripts": ["modbot-train = modbot.training.__main__:main",
                              "modbot = modbot.__main__:main"]},
      install_requires=['nltk', 'gensim', 'scikit-learn>=1.0.1', 'symspellpy',
                        'notify_run', 'pandas', 'sphinx-argparse',
                        'numpy>=1.20.3', 'dask_ml',
                        'dask', 'requests', 'websockets', 'emoji', 'pytz',
                        'googletrans', 'tqdm', 'joblib',
                        'scipy>=1.7.3', 'tensorflow>=2.7', 'keras',
                        'tensorflow_hub', 'tensorflow_text', 'torch',
                        'transformers'])
