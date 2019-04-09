Multi-Robot RMPflow
===================================================

This repository contains a python implementation of Multi-Robot RMPflow [[2]](https://arxiv.org/abs/1902.05177) and a 2D example for RMPflow [[1]](https://arxiv.org/abs/1811.07049).

## Prerequisites
+ Python3
+ Numpy
+ Scipy
+ Matplotlib
+ [The Robotarium python simulator](https://github.com/robotarium/robotarium_python_simulator) (optional, only required for `formation_preservation.py` and `cyclic_pursuit_formation.py`)


## To Run in simulation
2D example in [1]:

`python3 rmp_example.py`

A simple multi-robot go-to-goal example (centralized version):

`python3 multi_agent_rmp_centralized.py`


A simple multi-robot go-to-goal example (decentralized version):

`python3 multi_agent_rmp.py`

Formation preservation experiment in [2]: (the Robotarium python simulator required)

`python3 formation_preservation.py`

Cyclic pursuit experiment in [2]: (the Robotarium python simulator required)

`python3 cyclic_pursuit_formation.py`

## To Run on [the Robotarium](https://www.robotarium.gatech.edu/)

[The Robotarium](https://www.robotarium.gatech.edu/) is a remotely accessible swarm robotics testbed. The Robotarium gives users the chance to execute the same code they developed in simulation on real robots. A user can conveniently run an experiment on the Robotarium's robots remotely through the Robotarium's web interface. Please follow the instructions on [the Robotarium webpage](https://www.robotarium.gatech.edu/) to run the code on the real robots.

## References
+ [1] [RMPflow: A Computational Graph for Automatic Motion Policy Generation](https://arxiv.org/abs/1811.07049)
+ [2] [Multi-Objective Policy Generation for Multi-Robot Systems Using Riemannian Motion Policies](https://arxiv.org/abs/1902.05177)
+ [3] [The Robotarium: A remotely accessible swarm robotics research testbed](https://ieeexplore.ieee.org/document/7989200)


## Questions & Bug reporting

Please use Github issue tracker to report bugs. For other questions please contact [Anqi Li](mailto:anqi.li@gatech.edu).

## Citing

If you use the repository in an academic context, please cite following publications:

```
@article{li2019multi,
  title={Multi-Objective Policy Generation for Multi-Robot Systems Using Riemannian Motion Policies},
  author={Li, Anqi and Mukadam, Mustafa and Egerstedt, Magnus and Boots, Byron},
  journal={arXiv preprint arXiv:1902.05177},
  year={2019}
}


@inproceedings{cheng2018rmpflow,
  title={{RMP}flow: A Computational Graph for Automatic Motion Policy Generation},
  author={Cheng, Ching-An and Mukadam, Mustafa and Issac, Jan and Birchfield, Stan and Fox, Dieter and Boots, Byron and Ratliff, Nathan},
  booktitle ={The 13th International Workshop on the Algorithmic Foundations of Robotics},
  url={arXiv preprint arXiv:1811.07049},
  year={2018}
}
```
If you use the Robotarium, please cite following publication:
```
@inproceedings{pickem2017robotarium,
  title={The robotarium: A remotely accessible swarm robotics research testbed},
  author={Pickem, Daniel and Glotfelter, Paul and Wang, Li and Mote, Mark and Ames, Aaron and Feron, Eric and Egerstedt, Magnus},
  booktitle={2017 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1699--1706},
  year={2017},
  organization={IEEE}
}
```

## License

This repository is released under the BSD license, reproduced in the file LICENSE in this directory.
