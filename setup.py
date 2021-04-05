import setuptools

with open("longdescription.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dataturehub",
    version="0.1.0",
    author="Ian Duncan",
    author_email="ian@datature.io",
    description="Loader for models trained on the Datature platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datature/hub",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.7, <3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    install_requires=['tensorflow==2.3.0', 'requests>=2.25.1'])