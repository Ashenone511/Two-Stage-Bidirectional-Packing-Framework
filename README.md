# A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems

The increasing demands of modern logistics have driven the need for the development of efficient packing methods capable of addressing the 3D Packing Problem (3D-PP). While Deep Reinforcement Learning (DRL) has emerged as a promising solution, the conventional three-stage scheme used in existing DRL-based methods still faces challenges, particularly in coordinating the behaviors of its constituent sub-networks and managing the large action space for item placement on instances involving large-sized bins. This work proposes a two-stage scheme to integrate item index and orientation selections into a single sub-stage, thereby simplifying behavioral coordination. To mitigate the issue of excessive memory usage associated with the selection integration, a Set Transformer with Induced Set Attention Block (ISAB) is employed to encode the rotated item state, thus keeping relatively light computation. Additionally, we propose a bidirectional packing method that compresses the placement action space while encouraging the agent to explore reasonable placement positions.

<img src="https://github.com/Ashenone511/Two-Stage-Bidirectional-Packing-Framework/blob/main/fig/TS-BP.png" width=700>


## Paper
Our paper, **A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems**, is accepted at *IEEE Transactions on Automation Science and Engineering*.

lf our work is helpful for your research, please cite our paper:
```
@ARTICLE{11153660,
  author={Yin, Hao and Chen, Fan and He, Hongjie},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  month={},
  doi={10.1109/TASE.2025.3607410}}
```

## Requirements
Python 3.7+ <br>
Pytorch 1.12+

## Training
Run `main.py` to train the TS-BP framework.


