from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Package for inferring latent calcium activity from two-channel imaging'

# Setting up
setup(
    name="calcium_inference",
    version=VERSION,
    author="Matthew S. Creamer",
    author_email="matthew.s.creamer@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      ],

    keywords=['python', 'calcium', 'inference', 'two-channel', 'imaging'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
