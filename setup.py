"""BluePyEfe setup """

"""
Copyright (c) 2020, EPFL/Blue Brain Project

 This file is part of BluePyEfe <https://github.com/BlueBrain/BluePyEfe>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import setuptools
import versioneer

setuptools.setup(
    name='bluepyefe',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Blue Brain Python E-feature extraction',
    packages=setuptools.find_packages(),
    author="BlueBrain Project, EPFL",
    license="LGPLv3",
    keywords=(
        'neuroscience',
        'BlueBrainProject'),
    url='https://github.com/BlueBrain/BluePyEfe',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: GNU Lesser General Public '
        'License v3 (LGPLv3)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities'],
    install_requires=[
        'igor',
        'neo',
        'matplotlib',
        'efel',
        'sh',
        'pandas',
        'scipy'],
)
