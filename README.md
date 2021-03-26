# IntPrim
IntPrim is a Python Library for Interaction Primitives developed at the Interactive Robotics Lab at Arizona State University. Interaction Primitives (Ben Amor, 2014) provide a machine learning framework that allows users to learn controllers for human-robot interaction (HRI) from example demonstrations alone. This framework enables imitation learning in HRI settings in which two or more agents are physically interacting with each other. The overall objective is to extract the dynamics of an interaction from a set of example demonstrations, then use the dynamics to infer the future states of both the human and the robot. As a result IntPrim allows for a.) the prediction of future motions of a human partner, b.) the generation of appropriate robot reponses, and c.) the inference of latent (unobserved) variables that may be critical for the interaction. 

This library was primarily developed to enable training and inference with Bayesian Interaction Primitives (BIP), however, it also supports Probabilistic Movement Primitives and Particle Filters as a baseline for comparison.

| | |
|-|-|
| ![](docs/notebooks/media/catching_lq.gif?raw=true) | ![](docs/notebooks/media/hugging_lq.gif?raw=true) |
| ![](docs/notebooks/media/walking_lq.gif?raw=true) | ![](docs/notebooks/media/box_lq.gif?raw=true) |

This library has been successfully deployed in real-world HRI scenarios (some of which are shown above) involving cooperative object manipulation, shaking hands, hugging, grasping, and more!
A list of peer-reviewed publications that have utilized this library can be found below.

This library was created by Joseph Campbell, Simon Stepputtis, Michael Drolet, and Heni Ben Amor from Arizona State University.

A corresponding ROS package is available at https://github.com/ir-lab/intprim_framework_ros which greatly simplifies the process of setting up and running real experiments with the IntPrim library.

## Features

* Train an Interaction Primitives model from demonstrations
* Perform recursive inference with an Interaction Primitives model and generate future states
* Support for inference with Ensemble Bayesian Interaction Primitives, Bayesian Interaction Primitives, Probabilistic Movement Primitives with DTW, and Particle Filter
* Automatic basis space selection with support for Gaussian, Sigmoidal, and Polynomial functions
* Automatic computation of the observation noise
* Optimize interactions with Model Predictive Control scheme
* Comprehensive interactive analysis tools
* Integration with ROS via https://github.com/ir-lab/intprim_framework_ros

![](docs/notebooks/media/analysis_example.gif?raw=true)

## Tutorials and Documentation

A set of tutorials and documentation about IntPrim has been provided in the following Jupyter Notebooks:

1. [Introduction](docs/notebooks/1_introduction.ipynb)
2. [Quickstart](docs/notebooks/2_quickstart.ipynb)
3. [In-depth Tutorial](docs/notebooks/3_indepth_tutorial.ipynb)
4. [Mathematical Details](docs/notebooks/4_mathematical_details.ipynb)
5. [Model Predictive Control with Interaction Primitives](docs/notebooks/5_optimizing_interactions.ipynb)

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

python setup.py install --user

## Feedback
Questions or comments may be directed to Joseph Campbell at <jacampb1@asu.edu>, Simon Stepputtis at <sstepput@asu.edu>, Geoffrey Clark at <gmclark1@asu.edu>, or Heni Ben Amor <hbenamor@asu.edu>.

http://interactive-robotics.engineering.asu.edu/interaction-primitives/


## Further Information
For further information, please consult the below peer-reviewed conference papers:

| | | |
|-|-|-|
| <a href="https://arxiv.org/pdf/1908.04955.pdf">![](docs/notebooks/media/joe_rss19.png?raw=true)</a> | <a href="https://arxiv.org/pdf/1908.05552">![](docs/notebooks/media/joe_iros19.png?raw=true)</a> | <a href="http://proceedings.mlr.press/v78/campbell17a/campbell17a.pdf">![](docs/notebooks/media/joe_corl17.png?raw=true)</a> |

We have also made several conference presentations available to watch:

| | |
|-|-|
| <a href="https://youtu.be/vgkxR9TDqhY?t=9913">![](docs/notebooks/media/joe_rss19_play.png?raw=true)</a> | <a href="https://youtu.be/_9Ny2ghjwuY?t=26862">![](docs/notebooks/media/joe_corl17_play.png?raw=true)</a> |
| <a href="https://drive.google.com/open?id=1b6csa9OnmF7gL3DOy7d8WeMYh3YXA0hF">![](docs/notebooks/media/joe_icra20_play.png?raw=true)</a> | <a href="https://youtu.be/DxQPF3VwuoA">![](docs/notebooks/media/geoffrey_corlplay1.png?raw=true)</a> |


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

This project is licensed under the MIT license, included in this directory.

## Used By

This library has been developed by Joseph Campbell at Arizona State University with contributions from Simon Stepputtis, Geoffrey Clark, Michael Drolet, and Heni Ben Amor.
This library (or variations thereof) has been successfully utilized in the following works:

[J. Campbell and K. Yamane. Learning Whole-Body Human-Robot Haptic Interaction in Social Contexts. International Conference on Robotics and Automation (ICRA) 2020.](https://arxiv.org/pdf/2005.12508.pdf)

[J. Campbell, S. Stepputtis, and H. Ben Amor. Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks. Robotics: Science and Systems (RSS) 2019.](https://arxiv.org/pdf/1908.04955.pdf)

[J. Campbell, A. Hitzmann, S. Stepputtis, S. Ikemoto, K. Hosoda, and H. Ben Amor. Learning Interactive Behaviors for Musculoskeletal Robots Using Bayesian Interaction Primitives. International Conference on Intelligent Robots and Systems (IROS) 2019.](https://arxiv.org/pdf/1908.05552.pdf)

[K. Bagewadi, J. Campbell, and H. Ben Amor. Multimodal Dataset of Human-Robot Hugging Interaction. AAAI Fall Symposium on Artificial Intelligence for Human-Robot Interaction (AI-HRI), November 2019.](https://arxiv.org/pdf/1909.07471.pdf)

[J. Campbell and H. Ben Amor. Bayesian Interaction Primitives: A SLAM Approach to Human-Robot Interaction. Conference on Robot Learning (CoRL) 2017.](http://proceedings.mlr.press/v78/campbell17a/campbell17a.pdf)

[G. Clark, J. Campbell, S.M.R. Sorkhabadi, W. Zhang, and H. Ben Amor. Predictive Modeling of Periodic Behavior for Human-Robot Symbiotic Walking](https://arxiv.org/pdf/2005.13139.pdf)

[G. Clark, J. Campbell, and H. Ben Amor. Learning Predictive Models for Ergonomic Control of Prosthetic Devices](https://arxiv.org/pdf/2011.07005.pdf)

## Acknowledgements

This work was supported in part by the National Science Foundation under grant No. IIS-1749783 and the Honda Research Institute.

![](docs/notebooks/media/acknowledgement_logos.png?raw=true)
