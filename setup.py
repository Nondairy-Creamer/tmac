from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Two-channel motion artifact correction'

# Setting up
setup(
    name="tmac",
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

    keywords=['python', 'calcium', 'inference', 'two-channel', 'imaging', 'motion', 'correction'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
