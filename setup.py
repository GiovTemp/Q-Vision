# Q-Vision/setup.py
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Definizione delle estensioni Cython
extensions = [
    Extension(
        "qvision.remove_dead_time",                 # Nome del modulo
        sources=["qvision/remove_dead_time.pyx"],   # Percorso al file .pyx
        include_dirs=[numpy.get_include()],          # Directory di inclusione per NumPy
    )
]

setup(
    name='qvision',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'tabulate',
        'numba',
        'cython'
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
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",          # Usa Python 3
            'boundscheck': False,           # Disabilita i controlli degli indici per migliorare le prestazioni
            'wraparound': False,            # Disabilita il wraparound degli indici
            'initializedcheck': False,      # Disabilita il controllo dell'inizializzazione
            'nonecheck': False,             # Disabilita il controllo per None
        },
        annotate=True,                        # Genera un file HTML con le annotazioni Cython (utile per il debug)
    ),
    include_package_data=True,                # Includi i dati del pacchetto come i file Cython compilati
    zip_safe=False,                           # Non usare zip
)