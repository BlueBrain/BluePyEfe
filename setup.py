import setuptools

import versioneer

setuptools.setup(
    name='BluePyEfe',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='bluepyefe',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'igor',
        'neo',
        'matplotlib',
        'efel',
        'sh',
        'pandas==0.23.1',
        'scipy'],
)
