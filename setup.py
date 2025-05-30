"""
Script de instalación para Optimatrading
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optimatrading",
    version="1.0.0",
    author="Alejandro Del angel",
    author_email="aalejandro0406@gmail.com",
    description="Sistema avanzado de análisis de trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/optimatrading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "ccxt>=2.0.0",
        "redis>=4.0.0",
        "pyyaml>=5.4.0",
        "structlog>=21.1.0",
        "prometheus-client>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
) 