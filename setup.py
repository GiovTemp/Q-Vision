# Q-Vision/setup.py
from setuptools import setup, find_packages

setup(
    name='qvision',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    url='https://github.com/GiovTemp/Q-Vision',
    license='MIT',
    author='Giovanni Tempesta',
    author_email='g.tempesta16@studenti.uniba.it',
    description='A Python library for applying computer vision techniques to quantum phenomena',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)