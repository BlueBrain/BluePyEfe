import setuptools

setuptools.setup(
    name='BluePyEfe',
    version='0.1.2dev',
    description='bluepyefe',
    packages=setuptools.find_packages(),
    install_requires=[
        'igorpy',
        'neo',
        'matplotlib',
        'efel',
        'sh',
        'pandas',
        'scipy'],
)
