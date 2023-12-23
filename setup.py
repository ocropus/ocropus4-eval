from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req_file:
        return req_file.read().splitlines()

setup(
    name='ocroeval',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'ocrtexteval=ocroeval.eval:main',
            'runocr=ocroeval.runocr:main',
        ],
    },
)