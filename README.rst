LapTrack
========

|PyPI| |Status| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black| |Zenodo|

.. |PyPI| image:: https://img.shields.io/pypi/v/laptrack.svg
   :target: https://pypi.org/project/laptrack/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/laptrack.svg
   :target: https://pypi.org/project/laptrack/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/laptrack
   :target: https://pypi.org/project/laptrack
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/laptrack
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/laptrack/latest.svg?label=Read%20the%20Docs
   :target: https://laptrack.readthedocs.io/
   :alt: Read the documentation at https://laptrack.readthedocs.io/
.. |Tests| image:: https://github.com/yfukai/laptrack/workflows/Tests/badge.svg
   :target: https://github.com/yfukai/laptrack/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/yfukai/laptrack/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yfukai/laptrack
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5519538.svg
   :target: https://doi.org/10.5281/zenodo.5519538
   :alt: Zenodo

Features
--------

Provides a robust particle tracking algorithm using the Linear Assignment Problem, with various cost functions for linking.

Installation
------------

You can install *LapTrack* via pip_ from PyPI_:

.. code:: console

   $ pip install laptrack


Usage
-----

Please see the Usage_ for details.

Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `The 3-Clause BSD License`_,
*LapTrack* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

- This program implements a modified version of the algorithm in the `K. Jaqaman et al. (2008)`_.

- Inspired by TrackMate_ a lot. See documentation_ for its detailed algorithm, the `2016 paper`_, and the `2021 paper`_.

- This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.


Citation
--------

If you use this program for your research, please cite it and help us build more.

.. code-block:: bib

   @misc{laptrack,
      author = {Yohsuke T. Fukai},
      title = {laptrack},
      year  = {2021},
      url   = {https://doi.org/10.5281/zenodo.5519537},
   }


.. _K. Jaqaman et al. (2008): https://www.nature.com/articles/nmeth.1237
.. _TrackMate: https://imagej.net/plugins/trackmate/
.. _documentation: https://imagej.net/plugins/trackmate/algorithms
.. _2016 paper: https://doi.org/10.1016/j.ymeth.2016.09.016
.. _2021 paper: https://doi.org/10.1101/2021.09.03.458852

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _The 3-Clause BSD License: https://opensource.org/licenses/BSD-3-Clause
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/yfukai/laptrack/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://laptrack.readthedocs.io/en/latest/usage.html
