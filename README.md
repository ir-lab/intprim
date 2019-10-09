# IntPrim
The IntPrim library is a Python implementation of Interaction Primitives from the Interactive Robotics Lab at Arizona State University.
Interaction Primitives are a human-robot interaction (HRI) framework based on imitation learning.
The objective of this framework is to extract the dynamics of an interaction from a set of example demonstrations, then use the dynamics to infer the future states of both the human and the robot.
The primary purpose of this library is to enable training and inference using Bayesian Interaction Primitives, however, it also supports Probabilistic Movement Primitives and Particle Filters as a baseline for comparison.

![](docs/notebooks/media/examples_new.png?raw=true)

This library has been successfully deployed in real-world HRI scenarios involving cooperative object manipulation, shaking hands, hugging, grasping, and more!
A list of peer-reviewed publications that have utilized this library can be found below.

## Features

* Train a BIP model from demonstrations
* Perform recursive inference with an Interaction Primitives model and generate future states
* Support for inference with Ensemble Bayesian Interaction Primitives, Bayesian Interaction Primitives, Probabilistic Movement Primitives with DTW, and Particle Filter
* Automatic basis space selection with support for Gaussian, Sigmoidal, and Polynomial functions
* Automatic computation of the observation noise
* Comprehensive interactive analysis tools

![](docs/notebooks/media/analysis_example.gif?raw=true)

## Tutorials and Documentation

A set of tutorials and documentation about IntPrim has been provided in the following Jupyter Notebooks:

1. [Introduction](docs/notebooks/1_introduction.ipynb)
2. [Quickstart](docs/notebooks/2_quickstart.ipynb)
3. [In-depth Tutorial](docs/notebooks/3_indepth_tutorial.ipynb)
4. [Mathematical Details](docs/notebooks/4_mathematical_details.ipynb)

Additionally, the API and associated documentation can be found here:

https://ir-lab.github.io/intprim/

## Prerequisites

This library has been built and tested on Python 2.7.

The following Python libraries must be installed before IntPrim can be used:

* Numpy
* Scipy
* Matplotlib
* Sklearn

## Installation
To install this library, download the package and in the root directory run:

python setup.py build_ext install --user

This project is licensed under the MIT license, included in this directory.

## Feedback
Questions or comments may be directed to Joseph Campbell at <jacampb1@asu.edu>, Simon Stepputtis at <sstepput@asu.edu>, or Heni Ben Amor <hbenamor@asu.edu>.

http://interactive-robotics.engineering.asu.edu/interaction-primitives/

## Citation
If you use this library, please cite one of the following papers:
```
@InProceedings{campbell2019probabilistic,
  title={Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks},
  author={Campbell, Joseph and Stepputtis, Simon and Ben Amor, Heni},
  booktitle={Robotics: Science and Systems},
  year={2019}
}
```
```
@InProceedings{campbell19bayesian,
  title = {Bayesian Interaction Primitives: A SLAM Approach to Human-Robot Interaction},
  author = {Joseph Campbell and Heni Ben Amor},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {379--387},
  year = {2017},
  editor = {Sergey Levine and Vincent Vanhoucke and Ken Goldberg},
  volume = {78},
  series = {Proceedings of Machine Learning Research},
  address = {},
  month = {13--15 Nov},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v78/campbell17a/campbell17a.pdf},
  url = {http://proceedings.mlr.press/v78/campbell17a.html}
}
```

## Used By

This library has been developed by Joseph Campbell at Arizona State University and has been utilized in the following works:

[1] J. Campbell, S. Stepputtis, and H. Ben Amor. Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks. Robotics: Science and Systems (RSS) 2019.

[2] J. Campbell and H. Ben Amor. Bayesian Interaction Primitives: A SLAM Approach to Human-Robot Interaction. Conference on Robot Learning (CoRL) 2017.

[3] J. Campbell, A. Hitzmann, S. Stepputtis, S. Ikemoto, K. Hosoda, and H. Ben Amor. Learning Interactive Behaviors for Musculoskeletal Robots Using Bayesian Interaction Primitives. International Conference on Intelligent Robots and Systems (IROS) 2019.

[4] K. Bagewadi, J. Campbell, and H. Ben Amor. Multimodal Dataset of Human-Robot Hugging Interaction. AAAI Fall Symposium on Artificial Intelligence for Human-Robot Interaction (AI-HRI), November 2019.
