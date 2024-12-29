#! /usr/bin/env python
"""A template."""

import codecs
from setuptools import find_packages, setup


DISTNAME = 'keras_ns'
DESCRIPTION = ''
with codecs.open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = ''
MAINTAINER_EMAIL = ''
URL = ''
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = ''
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'tensorflow', 'problog', 'matplotlib', 'pandas', 'tensorflow_ranking']
CLASSIFIERS = []

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version="0.0.1",
      download_url=DOWNLOAD_URL,
      long_description_content_type='text/x-rst',
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES)
