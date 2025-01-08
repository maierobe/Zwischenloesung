from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="ProKI Hackathon 2024 Template",
    version="1.0.0",
    author="Max Goebels",
    description="ProKI Hackathon 2024 Example Project",
    install_requires=[req for req in requirements if req[:2] != "# "],
    packages=find_packages(),
)
