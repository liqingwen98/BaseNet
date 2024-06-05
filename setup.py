from setuptools import setup, find_packages

setup(
    name='basenet',
    packages = find_packages(),
    version='1.0',
    description='BaseNet: A Transformer-Based Toolkit for Nanopore Sequencing Signal Decoding',
    author='Qingwen Li',
    author_email='li.qing.wen@foxmail.com',
    url='https://github.com/liqingwen98/BaseNet.git',
    install_requires=[
        'funasr==0.7.0',
        'torch',
        'transformers==4.27.1'
    ],
    keywords=['nanopore sequencing', 'basecalling', 'transformer', 'end-to-end'],
    python_requires='>=3.8'
)
