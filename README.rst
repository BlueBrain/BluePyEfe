|banner|

-----------------

BluePyEfe: Blue Brain Python E-feature extraction
=================================================

+----------------+------------+
| Latest Release | |pypi|     |
+----------------+------------+
| Documentation  | |docs|     |
+----------------+------------+
| License        | |license|  |
+----------------+------------+
| Build Status 	 | |tests|    |
+----------------+------------+
| Coverage       | |coverage| |
+----------------+------------+
| Citation       | |zenodo|   |
+----------------+------------+
| Gitter         | |gitter|   |
+----------------+------------+


Introduction
============

BluePyEfe aims at easing the process of reading experimental recordings and extracting
batches of electrical features from these recordings. To do so, it combines trace reading
functions and features extraction functions from the `eFel library <https://github.com/BlueBrain/eFEL>`_.

BluePyEfe outputs protocols and features files in the format used
by `BluePyOpt <https://github.com/BlueBrain/BluePyOpt>`_ for neuron electrical model building.

How to cite
===========
If you use this repository in your work, please refer to the "Cite this repository" button at the top of the repository page to get various citation formats, including APA and BibTeX.

For detailed citation information, please refer to the `CITATION.cff <./CITATION.cff>`_ file.

Requirements
============

* `Python 3.9+ <https://www.python.org/downloads/release/python-380/>`_
* `eFEL eFeature Extraction Library <https://github.com/BlueBrain/eFEL>`_ (automatically installed by pip)
* `Numpy <http://www.numpy.org>`_ (automatically installed by pip)
* `Scipy <https://www.scipy.org/>`_ (automatically installed by pip)
* `Neo <https://neo.readthedocs.io/en/stable/>`_ (automatically installed by pip)
* The instruction below are written assuming you have access to a command shell on Linux / UNIX / MacOSX / Cygwin

Installation
============

To install BluePyEfe, run:

.. code-block:: bash

    pip install bluepyefe


Quick Start and Operating Principle
===================================

For a hands-on introduction to BluePyEfe, have a look at the notebook `examples/example_of_extraction.ipynb <examples/example_of_extraction.ipynb>`_

The goal of the present package is to extract meaningful electrophysiological features (e-features) from voltage time series.
The e-features considered in the present package are the one implemented in the `eFEL python library <https://github.com/BlueBrain/eFEL>`_. See `this pdf <https://bluebrain.github.io/eFEL/efeature-documentation.pdf>`_ for a list of available e-features.

The present package makes one major assumption: E-features are more meaningful if they are coming from a set of traces rather than a single trace. And they are even more meaningful if these traces come from different cells of the same cellular type.
This assumption dictates the organisation of the package and has several consequences:

The efeatures extracted through the package will always be averaged over the trace considered. For example, the AP_amplitude will be an average over all the action potentials present in a trace. If you wish to work on an AP by AP basis, please consider using the eFEL library directly.

A large part of the present software is therefore dedicated to averaging the features across set of "equivalent" recordings. To be able to average e-features across different cells in a meaningful way, an equivalence must be established between the traces coming from these different cells. It would not make sense to average the mean firing frequency obtain cell A on a 1s long step protocol with the one obtain for cell B on a ramp protocol that lasts for 500ms. We chose to define recordings as equivalent based on two criteria: (1) They have the same name and (2) they are of the same amplitude when the amplitude is expressed as a percentage of the rheobase of the cell.

A pseudo-code for the main function of the package (bluepyefe.extract.extract_efeatures) could look as follows:

#. Load the data to memory by reading all the files containing the traces
#. Extract the required e-features for all the traces
#. Compute the rheobases of the cells based on one or several protocols
#. Use these rheobases to associate to each protocol an amplitude expressed in % of the rheobase
#. Compute the mean and standard deviations for the e-features across traces having the same amplitude
#. Save the results and plot the traces and e-features

Each of these steps are parametrized by a number of settings, therefore we recommend that you read carefully the docstring of the function.

Coming from the legacy version
==============================
The legacy version (v0.4*) is moved to the legacy branch.
Changes introduced in v2.0.0 are listed in the `CHANGELOG.rst <CHANGELOG.rst>`_.
That is the only file you need to look at for the changes as the future changes will also be noted there.

Funding
=======
This work has been partially funded by the European Union Seventh Framework Program (FP7/2007­2013) under grant agreement no. 604102 (HBP), and by the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 720270 (Human Brain Project SGA1) and No. 785907 (Human Brain Project SGA2) and by the EBRAINS research infrastructure, funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

Copyright (c) 2016-2024 Blue Brain Project/EPFL

.. |pypi| image:: https://img.shields.io/pypi/v/bluepyefe.svg
               :target: https://pypi.org/project/bluepyefe/
               :alt: latest release
.. |docs| image:: https://readthedocs.org/projects/bluepyefe/badge/?version=latest
               :target: https://bluepyefe.readthedocs.io/
               :alt: latest documentation
.. |license| image:: https://img.shields.io/pypi/l/bluepyefe.svg
                  :target: https://github.com/BlueBrain/bluepyefe/blob/master/LICENSE.txt
                  :alt: license
.. |tests| image:: https://github.com/BlueBrain/BluePyEfe/workflows/Build/badge.svg?branch=master
                :target: https://github.com/BlueBrain/BluePyEfe/actions
                :alt: Actions build status
.. |coverage| image:: https://codecov.io/github/BlueBrain/BluePyEfe/coverage.svg?branch=master
                   :target: https://codecov.io/gh/BlueBrain/BluePyEfe
                   :alt: coverage
.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
                 :target: https://gitter.im/bluebrain/bluepyefe
                 :alt: gitter
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3728191.svg
                 :target: https://doi.org/10.5281/zenodo.3728191
                 :alt: DOI

..
    The following image is also defined in the index.rst file, as the relative path is
    different, depending from where it is sourced.
    The following location is used for the github README
    The index.rst location is used for the docs README; index.rst also defined an end-marker,
    to skip content after the marker 'substitutions'.

.. substitutions
.. |banner| image::  https://raw.githubusercontent.com/BlueBrain/BluePyEfe/master/docs/source/logo/BluePyEfeBanner.jpg

