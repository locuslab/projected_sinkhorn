from setuptools import find_packages, setup

setup(
    name='projected_sinkhorn',
    version='0.2',
    description="A library implementing the projected sinkhorn iteration for PyTorch",
    author='Eric Wong',
    author_email='ericwong@cs.cmu.edu',
    platforms=['any'],
    license="MIT",
    url='https://github.com/locuslab/projected_sinkhorn',
    packages=['projected_sinkhorn'],
    install_requires=[
        'torch==1.0'
    ]
)
