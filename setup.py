#!/usr/bin/env python3
"""
Setup script for Sigray Machine Learning Platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="sigray-ml-platform",
    version="1.0.0",
    author="Sigray, Inc.",
    author_email="support@sigray.com",
    description="Advanced 3D image enhancement platform using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sigray/ml-platform",
    project_urls={
        "Bug Reports": "https://github.com/sigray/ml-platform/issues",
        "Source": "https://github.com/sigray/ml-platform",
        "Documentation": "https://github.com/sigray/ml-platform/wiki",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "tensorboard": [
            "tensorboard>=2.0",
            "tensorboardX>=2.0",
        ],
        "monitoring": [
            "psutil>=5.0",
            "GPUtil>=1.4",
        ],
        "gui": [
            "tkinter",
            "pillow>=8.0",
            "matplotlib>=3.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "tensorboard>=2.0",
            "tensorboardX>=2.0",
            "psutil>=5.0",
            "GPUtil>=1.4",
            "tkinter",
            "pillow>=8.0",
            "matplotlib>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sigray-train=cli.train_cli:main",
            "sigray-infer=cli.inference_cli:main",
            "sigray-gui=gui.main_window:main",
            # Legacy aliases
            "3d-enhance-train=cli.train_cli:main",
            "3d-enhance-infer=cli.inference_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning",
        "image processing",
        "3d imaging",
        "tiff",
        "u-net",
        "pytorch",
        "sigray",
        "scientific imaging",
        "image enhancement",
    ],
)