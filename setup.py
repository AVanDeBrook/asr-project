from setuptools import setup, find_packages

setup(
    name="asr_project",
    version="1.0.0",
    description="A collection of tools for data processing and analysis and machine learning model training, inference, and application.",
    author="Aaron Van De Brook",
    author_email="avandebrook5@gmail.com",
    url="https://avandebrook.github.io/asr-project/",
    package_dir={"": "source"},
    packages=find_packages(where="source"),
    install_requires=[
        "torch",
        "pandas",
        "openpyxl",
        "librosa",
        "pytorch_lightning",
        "matplotlib",
        "Cython",
        "nemo_toolkit[all]",
    ],
)
