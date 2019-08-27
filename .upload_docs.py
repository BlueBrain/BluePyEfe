#!/bin/env python

from __future__ import print_function

import sys
import os
import contextlib
import datetime

metadata_template = \
    """---
packageurl: https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/bluepyefe
major: {major_version}
description: Get efeatures from experimental data
repository: https://bbpcode.epfl.ch/code/#/admin/projects/sim/BluePyEfe
externaldoc: https://bbpcode.epfl.ch/documentation/#BluePyEfe
updated: {date}
maintainers: Werner Van Geit
name: BluePyEfe
license: BBP-internal-confidential
version: {version}
contributors: Christian Roessert, Werner Van Geit,  BBP
minor: {minor_version}
---
"""


@contextlib.contextmanager
def cd(dir_name):
    """Change directory"""
    old_cwd = os.getcwd()
    os.chdir(dir_name)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def main():
    """Main"""
    doc_dir = sys.argv[1]

    doc_dir = os.path.abspath(doc_dir)

    with cd(doc_dir):
        print('Reading BluePyEfe version ...')
        import bluepyefe
        bluepyefe_version = bluepyefe.__version__
        bluepyefe_major_version = bluepyefe_version.split('.')[0]
        bluepyefe_minor_version = bluepyefe_version.split('.')[1]
        print('BluePyEfe version is: %s' % bluepyefe_version)

        finished_filename = '.doc_version'

        if os.path.exists(finished_filename):
            os.remove(finished_filename)

        metadata_filename = 'metadata.md'

        metadata_content = metadata_template.format(
            major_version=bluepyefe_major_version,
            minor_version=bluepyefe_minor_version,
            date=datetime.datetime.now().strftime("%d/%m/%y"),
            version=bluepyefe_version)

        print('Created metadata: %s' % metadata_content)

        with open(metadata_filename, 'w') as metadata_file:
            metadata_file.write(metadata_content)

        print('Wrote metadata to: %s' % metadata_filename)

        with open(finished_filename, 'w') as finished_file:
            finished_file.write(bluepyefe_version)

        print('Wrote doc version info to: %s' % finished_filename)


if __name__ == '__main__':
    main()
