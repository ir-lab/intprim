# intprim
The Interaction Primitives Python library from the Interactive Robotics Lab at Arizona State University.

This library implements the Bayesian Interaction Primitives algorithm.

## Installation
To install this library, download the package and in the root directory run:

python setup.py build_ext install --user

By default, this will make use of the included cythonized .c files. The .pyx files are included so that the files can be changed and re-compiled, and a vanilla Python implementation is included as well.

## Usage
To run the included examples, in a Python environment run:

```python
import intprim as ip
import intprim.examples

ip.examples.minimal()
ip.examples.spatial_robustness()
ip.examples.temporal_robustness()
```

The source code for the examples can be viewed under intprim/examples.

Note that in general, spatial noise and temporal noise are at odds with each other in conditioning.
If you have a lot of spatial noise it will be more difficult to accurately locate the correct phase, and vice versa.
So applications with lots of spatial noise AND lots of temporal noise may be challenging.

This project is licensed under the MIT license, included in this directory.

## Feedback
Questions or comments may be directed to Joseph Campbell at <jacampb1@asu.edu> or Simon Stepputtis at <sstepput@asu.edu>.

http://interactive-robotics.engineering.asu.edu/interaction-primitives/

## Citation
If you use this library, please use this citation:
```
@InProceedings{campbell17a,
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
