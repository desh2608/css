from setuptools import setup, find_packages

long_description = open("README.md").read()

setup(
    name="css",
    packages=find_packages(),
    version="0.1.0",
    license="MIT",
    description="Continuous speech separation with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    url="https://github.com/desh2608/torch-gss",
    keywords=["continuous speech separation"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
