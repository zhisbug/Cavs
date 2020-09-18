<p align="center"><img src="cavs_logo.png" width=256 /></p>


This project is an implementation of the Cavs system presented in the paper: [Cavs: An Efficient Runtime System for Dynamic Neural Networks, ATC'18](https://www.usenix.org/system/files/conference/atc18/atc18-xu-shizhen.pdf), sponsored by [Petuum Inc](https://petuum.com/).


## Introduction
Recent deep learning (DL) models are moving more and more to dynamic neural network (NN) architectures, where the NN structure changes for every data sample. 
However, existing DL programming models are inefficient in handling dynamic network architectures because of: 
- substantial overhead caused by repeating dataflow graph construction and processing every example; 
- difficulties in batched execution of multiple samples;
- inability to incorporate graph optimization techniques such as those used in static graphs. 

In this paper, we present **Cavs**, a runtime system that overcomes these bottlenecks and achieves efficient training and inference of dynamic NNs. 
Cavs represents a dynamic NN as a static vertex function <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{F}"> and a dynamic instance-specific graph <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{G}">. 
It avoids the overhead of repeated graph construction by only declaring and constructing <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{F}"> once, and allows for the use of static graph optimization techniques
on pre-defined operations in <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{F}">. 
Cavs performs training and inference by scheduling the execution of <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{F}"> following the dependencies in <img src="https://render.githubusercontent.com/render/math?math=\large \mathcal{G}">, hence naturally exposing batched execution opportunities over different samples.

Experiments comparing Cavs to state-of-the-art frameworks for dynamic NNs ([TensorFlow Fold](https://github.com/tensorflow/fold), [PyTorch](https://github.com/pytorch/pytorch) and [DyNet](https://github.com/clab/dynet)) demonstrate the efficacy of our approach: Cavs achieves a near one order of magnitude speedup on training of dynamic NN architectures, and ablations verify the effectiveness of our proposed design and optimizations.



## How to cite 
```
@inproceedings{xu2018cavs,
  title={Cavs: An efficient runtime system for dynamic neural networks},
  author={Xu, Shizhen and Zhang, Hao and Neubig, Graham and Dai, Wei and Kim, Jin Kyu and Deng, Zhijie and Ho, Qirong and Yang, Guangwen and Xing, Eric P},
  booktitle={2018 $\{$USENIX$\}$ Annual Technical Conference ($\{$USENIX$\}$$\{$ATC$\}$ 18)},
  pages={937--950},
  year={2018}
}
```
