"""
setup.py — pFedLLM installable package
Install with: pip install -e .
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pfedllm",
    version="1.0.0",
    author="pFedLLM Authors",
    description="LLM-Driven Personalized Federated Learning for Medical Imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pFedLLM",
    packages=find_packages(exclude=["tests*", "figures*", "outputs*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "rouge-score>=0.1.2",
        "evaluate>=0.4.0",
    ],
    extras_require={
        "dp":  ["opacus>=1.4.0"],
        "vis": ["umap-learn>=0.5.3", "seaborn>=0.12.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "pfedllm-train=train:main",
            "pfedllm-eval=evaluate:main",
            "pfedllm-demo=demo_numpy:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "federated learning", "personalized federated learning",
        "large language models", "medical imaging", "differential privacy",
        "radiology", "chest x-ray", "MIMIC-CXR",
    ],
)
