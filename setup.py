try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'My Project: my own machine learning library - SunML',
    'author': 'Jie Yang',
    'url': 'https://github.com/YoungGod/SunML/.',
    'download_url': 'https://github.com/YoungGod/SunML/.',
    'author_email': 'dearyangjie@yeah.net',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['SunML'],
    'scripts': [],
    'name': 'SunML'
}

setup(**config)
