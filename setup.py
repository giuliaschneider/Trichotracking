from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
]

setup(
    name='trichotracking',
    version='1.0',
    description='trichotracking finds and quantifies the movement of pairs of gliding filaments in image sequences.',
    author='Giulia Schneider',
    packages=find_packages(exclude=('test',)),
    install_requires=requirements
)
