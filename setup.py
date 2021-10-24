from setuptools import setup, find_packages

setup(
    name='facemask',
    version='0.1.0',
    packages=find_packages(include=['*', 'align', 'align.*', 'src', 'src.*']),
    install_requires=[
                        'scikit-learn',
                        'numpy',
                        'numba',
                        'opencv-python',
                        'dask[dataframe]',
                        'jupyter',
                        'filterpy',
                        'tensorflow-gpu',
                        'csv']
)
