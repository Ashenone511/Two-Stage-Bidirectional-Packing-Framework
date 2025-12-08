# Two-Stage Bidirectional Packing Framework

This repository contains the TS-BP framework code for solving three-dimensional packing problems. This framework combines a two-stage scheme with a bidirectional packing method to enhance space utilization in large-container packing problems.

## Requirements

This code is ran on a machine with Python version 3.8.0 and PyTorch 2.4.0.

## Usage
Run `main.py` to train the TS-BP framework. If you wish to change the item quantity and container sizes, modify the `nof_item`, `L`, and `W` values in `config.py`. 

If you wish to modify container sizes other than 100×100, 200×200, and 400×200, you must also modify the convolutional network in `Net.py`.

## Paper
Our paper, **A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems**, is accepted at *IEEE Transactions on Automation Science and Engineering*.

lf our work is helpful for your research, please cite our paper:
```
@ARTICLE{11153660,
  author={Yin, Hao and Chen, Fan and He, Hongjie},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems}, 
  year={2025},
  volume={22},
  number={},
  pages={21541-21551},
  doi={10.1109/TASE.2025.3607410}}

```




