from setuptools import find_packages, setup

setup(
    name="LFPAnalysis",
    version="1.0.0",
    description="Package to process LFP data",
    url="https://github.com/seqasim/LFPAnalysis",
    author="Salman E. Qasim and contributors",
    packages=find_packages(),
    package_data={"": ["data/*"]},
    include_package_data=True,
    python_requires=">=3.10",
)
