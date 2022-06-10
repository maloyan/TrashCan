from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="trash",
    packages=find_packages(),
    version="0.1.0",
    description="🗑️ classification",
    author="Narek Maloyan",
    license="MIT",
)