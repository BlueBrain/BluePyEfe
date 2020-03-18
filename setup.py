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
    description='bluepyefe',
    packages=setuptools.find_packages(),
    author="BlueBrain Project, EPFL",
    author_email="werner.vangeit@epfl.ch",
    license="LGPLv3",
    keywords=(
        'neuroscience',
        'BlueBrainProject'),
    url='https://github.com/BlueBrain/BluePyEfe',
    install_requires=[
        'igor',
        'neo',
        'matplotlib',
        'efel',
        'sh',
        'pandas',
        'scipy'],
)
