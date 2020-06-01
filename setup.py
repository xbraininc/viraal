# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setup(
        name="viraal",
        version="0.2.0",
        author="Badr Youbi Idrissi",
        author_email="badryoubiidrissi@gmail.com",
        description="Viraal package",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=["tests", "outputs", "multirun"]),
        classifiers=[
            # Feel free to choose another license
            "License :: OSI Approved :: MIT License",
            # Python versions are used by the noxfile in Hydra to determine which
            # python versions to test this plugin with
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Operating System :: OS Independent",
        ],
        install_requires=["hydra-core",
            "allennlp==0.9.0",
            "hydra-core==0.11.2",
            "hydra-range-sweeper-badr==0.1.1",
            "hydra-ray-launcher-badr==0.1.1",
            "ometrics==1.0.2",
            "pytest==5.3.1",
            "ray==0.8",
            "torch==1.2",
            "torchvision==0.4.2",
            "pandas",
            "umap"
         ]
    )
