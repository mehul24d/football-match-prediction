from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="football-match-prediction",
    version="0.1.0",
    author="mehul24d",
    description="Production-grade ML project for predicting football match outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mehul24d/football-match-prediction",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "fmp-train=src.models.train:main",
            "fmp-predict=src.models.predict:main",
            "fmp-api=api.main:start",
        ]
    },
)
