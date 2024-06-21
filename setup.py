import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hex_gym_env",
    version="1.0",
    description=("The game of Hex implemented for reinforcement learning in"
                 " the OpenAI gym framework."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MBPrdctns/hex_gym_env.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.2',
        'gym>=0.17.1',
        'gymnasium',
        "sb3_contrib",
        "pygame"
    ]
)
