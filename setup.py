import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="m2py",  # Replace with your own username
    version="0.0.3",
    author="Diego Torrejon",
    author_email="diego.torrejon.medina@gmail.com",
    description="Materials Morphology Python Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ponl/m2py",
    packages=["m2py"],
    install_requires=["matplotlib==2.2.3", "numpy==1.15.1", "scipy==1.1.0", "sklearn==0.0", "scikit-image==0.16.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
