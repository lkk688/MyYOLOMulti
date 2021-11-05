#python setup.py develop
from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "PIL", "matplotlib"]

setup(
    name="MyYOLOMulti",
    version="0.0.1",
    author="Kaikai Liu",
    author_email="kaikai.liu@sjsu.edu",
    description="A multiheader detector based on YOLO",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/lkk688/MyYOLOMulti",
    packages=find_packages(exclude=['tests', 'data', 'outputs']),
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)