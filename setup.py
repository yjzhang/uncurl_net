from setuptools import setup, find_packages

setup(name='uncurlnet',
        version='0.0.1',
        description='UncurlNet',
        author='Yue Zhang',
        author_email='yjzhang@cs.washington.edu',
        license='MIT',
        packages=find_packages('.'),
        install_requires=['numpy', 'scipy', 'scikit-learn', 'torch', 'uncurl-seq'],
        test_suite='nose.collector',
        test_requires=['nose', 'flaky'],
)
