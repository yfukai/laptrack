LapTrack
========

|PyPI| |Status| |Python Version| |License| |Download|

|Read the Docs| |Tests| |Codecov| |pre-commit| |Black|

|Publication| |Preprint| |Zenodo|

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
.. |Download| image:: https://img.shields.io/pepy/dt/laptrack
   :target: https://pypi.org/project/laptrack
   :alt: Total Download
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
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5519537.svg
   :target: https://doi.org/10.5281/zenodo.5519537
   :alt: Zenodo
.. |Publication| image:: https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtac799-167DA4
   :target: https://doi.org/10.1093/bioinformatics/btac799
   :alt: Bioinformatics
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-10.1101%2F2022.10.05.511038-bd2736
   :target: https://doi.org/10.1101/2022.10.05.511038
   :alt: bioRxiv

Features
--------

Provides a robust particle tracking algorithm using the Linear Assignment Problem, with various cost functions for linking.

See the `publication`_ and `associated repository`_ for the algorithm and parameter optimization by `Ray-Tune`_.

Requirements
------------

Python >= 3.8 is supported.
The software is tested against Python 3.8-3.12 in Ubuntu, and 3.12 in MacOS and Windows environments,
but the other combinations should also be fine. Please `file an issue`_ if you encounter any problem.

Installation
------------

You can install *LapTrack* via pip_ from PyPI_:

.. code:: console

   $ pip install laptrack

In Google Colaboratory, try

.. code:: console

   $ pip install --upgrade laptrack spacy flask matplotlib

to update the pre-installed packages.


Usage
-----

Please see the Usage_ for details.
The example notebooks are provided in `docs/examples <https://github.com/yfukai/laptrack/tree/main/docs/examples>`_.


================================= ============================================================================================ ======================
 notebook name                     short description                                                                            Google Colaboratory
--------------------------------- -------------------------------------------------------------------------------------------- ----------------------
 `api_example.ipynb`_              Introducing the package API by a simple example.                                               |colab|
--------------------------------- -------------------------------------------------------------------------------------------- ----------------------
 `bright_spots.ipynb`_             Application example: detecting bright spots by scikit-image `blob_log` and tracking them.
--------------------------------- -------------------------------------------------------------------------------------------- ----------------------
 `cell_segmentation.ipynb`_        Application example: tracking centroids of the segmented C2C12 cells undergoing divisions.
--------------------------------- -------------------------------------------------------------------------------------------- ----------------------
 `napari_interactive_fix.ipynb`_   Illustrates the usage of the ground-truth-preserved tracking with `napari`.
--------------------------------- -------------------------------------------------------------------------------------------- ----------------------
 `overlap_tracking.ipynb`_         Illustrates the usage of the custom metric to use segmentation overlaps for tracking.
================================= ============================================================================================ ======================

.. _api_example.ipynb:            https://github.com/yfukai/laptrack/tree/main/docs/examples/api_example.ipynb
.. _bright_spots.ipynb:           https://github.com/yfukai/laptrack/tree/main/docs/examples/bright_spots.ipynb
.. _cell_segmentation.ipynb:      https://github.com/yfukai/laptrack/tree/main/docs/examples/cell_segmentation.ipynb
.. _napari_interactive_fix.ipynb: https://github.com/yfukai/laptrack/tree/main/docs/examples/napari_interactive_fix.ipynb
.. _overlap_tracking.ipynb:       https://github.com/yfukai/laptrack/tree/main/docs/examples/overlap_tracking.ipynb

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
           :target: https://colab.research.google.com/github/yfukai/laptrack/blob/main/docs/examples/api_example.ipynb

The `API reference <https://laptrack.readthedocs.io/en/latest/reference.html>`_ covers the main classes and functions provided by LapTrack.

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
please `file an issue <https://github.com/yfukai/laptrack/issues>`_ along with a detailed description.


Credits
-------

- This program implements a modified version of the algorithm in the `K. Jaqaman et al. (2008)`_.

- Inspired by TrackMate_ a lot. See documentation_ for its detailed algorithm, the `2016 paper`_, and the `2021 paper`_.

- The data in `docs/examples/napari_interactive_fix_data` are generated by cropping images in `10.5281/zenodo.6087728 <https://doi.org/10.5281/zenodo.6087728>`_, which is distributed with `Creative Commons Attribution 4.0 International`_.

- The data in `docs/examples/cell_segmentation_data` are generated by cropping and resizing images in https://osf.io/ysaq2/, which is distributed with `Creative Commons Attribution 4.0 International`_. See `10.1038/sdata.2018.237 <https://doi.org/10.1038/sdata.2018.237>`_ for details.

- The data in `docs/examples/overlap_tracking_data` is generated by cropping `segmentation.npy` in https://github.com/NoneqPhysLivingMatterLab/cell_interaction_gnn, which is distributed with `Apache License 2.0`_. See the `original paper <https://doi.org/10.1371/journal.pcbi.1010477>`_ for details.

- The data in `docs/examples/3D_tracking_data` is generated by resizing iamges in https://bbbc.broadinstitute.org/BBBC050 , which is distributed with `Creative Commons Attribution 3.0 Unported License`_. See `10.1038/s41540-020-00152-8 <https://doi.org/10.1038/s41540-020-00152-8>`_ for details.

- This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.


Citation
--------

If you use this program for your research, please cite it and help us build more.

.. code-block:: bib

   @article{fukai_2022,
     title = {{{LapTrack}}: Linear Assignment Particle Tracking with Tunable Metrics},
     shorttitle = {{{LapTrack}}},
     author = {Fukai, Yohsuke T and Kawaguchi, Kyogo},
     year = {2022},
     month = dec,
     journal = {Bioinformatics},
     pages = {btac799},
     issn = {1367-4803},
     doi = {10.1093/bioinformatics/btac799},
   }

   @misc{laptrack,
      author = {Yohsuke T. Fukai},
      title = {laptrack},
      year  = {2021},
      url   = {https://doi.org/10.5281/zenodo.5519537},
   }

.. _publication: https://doi.org/10.1093/bioinformatics/btac799
.. _associated repository: https://github.com/NoneqPhysLivingMatterLab/laptrack-optimisation
.. _Ray-Tune: https://www.ray.io/ray-tune

.. _K. Jaqaman et al. (2008): https://www.nature.com/articles/nmeth.1237
.. _TrackMate: https://imagej.net/plugins/trackmate/
.. _documentation: https://imagej.net/plugins/trackmate/algorithms
.. _2016 paper: https://doi.org/10.1016/j.ymeth.2016.09.016
.. _2021 paper: https://doi.org/10.1101/2021.09.03.458852
.. _Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/legalcode
.. _Creative Commons Attribution 3.0 Unported License: https://creativecommons.org/licenses/by/3.0/legalcode
.. _The 3-Clause BSD License: https://opensource.org/licenses/BSD-3-Clause
.. _Apache License 2.0: https://opensource.org/licenses/Apache-2.0

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://laptrack.readthedocs.io/en/latest/usage.html
