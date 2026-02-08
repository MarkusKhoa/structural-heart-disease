"""
Setup script for the Structural Heart Disease Joint Embedding project.
This allows the project to be installed as a package in editable mode.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shd-joint-embedding",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Joint embedding model for ECG waveforms and tabular features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/structural_heart_disease",
    packages=find_packages(exclude=["tests", "docs", "notebooks", "echonext_dataset"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
            "ipywidgets>=8.0.0",
        ],
        "viz": [
            "tensorboard>=2.12.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shd-train=train:main",
        ],
    },
)
