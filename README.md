# intprim
The Interaction Primitives library from the Interactive Robotics Lab at Arizona State University.

This library implements Bayesian Interaction Primitives introduced in [1].

To install this library, download the package and in the root directory run:

python setup.py build_ext install --user

By default, this will make use of the included cythonized .c files. The .pyx files are included so that the files can be changed and re-compiled, and a vanilla Python implementation is included as well.

To run the included examples, in a Python environment run:

>>> import intprim as ip
>>> import intprim.examples
>>>
>>>
>>> ip.examples.minimal()
>>> ip.examples.spatial_robustness()
>>> ip.examples.temporal_robustness()

The source code for the examples can be viewed under intprim/examples.

Note that in general, spatial noise and temporal noise are at odds with each other in conditioning.
If you have a lot of spatial noise it will be more difficult to accurately locate the correct phase, and vice versa.
So applications with lots of spatial noise AND lots of temporal noise may be challenging.
Additionally, the included examples are atypical in that they only contain 2 DOF and one of them is removed during inference to simulate an interaction scenario in which we want to generate a response.
Thus the localization problem is based entirely on 1 DOF with no redundancies.

This project is licensed under the MIT license, included in this directory.

Questions or comments may be directed to Joseph Campbell at <jacampb1@asu.edu> or Simon Stepputtis at <sstepput@asu.edu>.

Cite: If you use this library, please use this citation:
@inproceedings{campbell2017bayesian,
    title={Bayesian Interaction Primitives: A SLAM Approach to Human-Robot Interaction},
    author={Campbell, Joseph and Amor, Heni Ben},
    booktitle={Conference on Robot Learning},
    pages={379–387},
    year={2017}
}
