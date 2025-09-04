# Two-Stage-Bidirectional-Packing-Framework
A Learning-Based Two-Stage Bidirectional Packing Framework for 3D Packing Problems

### Introduction
The increasing demands of modern logistics have driven the need for the development of efficient packing methods capable of addressing the 3D Packing Problem (3D-PP). While Deep Reinforcement Learning (DRL) has emerged as a promising solution, the conventional three-stage scheme used in existing DRL-based methods still faces challenges, particularly in coordinating the behaviors of its constituent sub-networks and managing the large action space for item placement on instances involving large-sized bins. This work proposes a two-stage scheme to integrate item index and orientation selections into a single sub-stage, thereby simplifying behavioral coordination. To mitigate the issue of excessive memory usage associated with the selection integration, a Set Transformer with Induced Set Attention Block (ISAB) is employed to encode the rotated item state, thus keeping relatively light computation. Additionally, we propose a bidirectional packing method that compresses the placement action space while encouraging the agent to explore reasonable placement positions. Experimental results demonstrate that the Two-Stage Bidirectional Packing (TS-BP) framework, formed by the above components, improves space utilization of 2.8%-4.6% on high-difficulty packing instances compared to current state-of-the-art methods.

### Requirements
Python 3.6+ <br>
Pytorch 1.12+

### Training
Run `main.py` to train the TS-BP framework.


