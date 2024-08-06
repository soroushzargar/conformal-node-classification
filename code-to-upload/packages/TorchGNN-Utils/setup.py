import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="tgnnu",
    version="3.0.4",

    author='Anonymous',
    author_email='Anonymous',
    url='Anonymous',

    description="Toolkits for easier work with PyTorch Geometric",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='License :: OSI Approved :: MIT License',
)
