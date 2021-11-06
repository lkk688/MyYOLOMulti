#python setup.py develop
#the setup.py develop command. It works very similarly to setup.py install, except that it doesn’t actually install anything. Instead, it creates a special .egg-link file in the deployment directory, that links to your project’s source code.
#When you’re done with a given development task, you can remove the project source from a staging area using setup.py develop --uninstall, specifying the desired staging area if it’s not the default.

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