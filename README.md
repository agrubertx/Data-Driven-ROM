# A  Comparison of Neural Network Architectures for Reduced-Order Modeling

This is an implementation of the algorithms from:

A. Gruber, M. Gunzburger, L. Ju, and Z. Wang. *A Comparison of Neural Network Architectures for Data-Driven Reduced-Order Modeling*, (under review).  Preprint [available here.](https://arxiv.org/abs/2110.03442#),

which compare various autoencoder architectures for PDE ROM applications.  In particular, we implement ROM autoencoders based on GCN2 graph convolutional (_GCN2), standard convolutional (_CNN), and fully connected (_FC) architectures.

The *src* folder contains the source code, and the examples from the paper are reproduced by running the files in the relevant *scripts* folder.  Network parameters can be changed at network initialization (see *src/networks*), and training parameters can be changed using the helper function "train_network" (see *src/utils*).


## Installation
The code was written and tested using Python 3.8 on Mac OSX 11.2.3.  The required dependencies are
* [Numpy](https://numpy.org/)
* [Pytorch](https://pytorch.org/)
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
* [Matplotlib](https://matplotlib.org/) (for visualization)

## Data
The necessary data for reproducing the experiments in the paper is found in the "datasets" folder.  Note that the larger files are stored using Git LFS due to their size.  To use your own dataset, load it into a data object with the following attributes:
- *xTrain, xValid* tensors representing training and validation solution snapshots.
- *pTrain, pValid* tensors representing training and validation points in parameter space, ordered consistently with the solution snapshots.
- *edge_index* (2 x nEdges) LongTensor encoding the sparse adjacency matrix (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)).
Take a look at the network classes in *src/networks* and the function "train_network" in *src/utils* for a better idea of what is used.

## Citation
Please cite [our paper](https://arxiv.org/pdf/2110.03442.pdf) if you use this code in your own work:
```
@misc{gruber2021comparison,
  title={A Comparison of Neural Network Architectures for Data-Driven Reduced-Order Modeling},
  author={Anthony Gruber and Max Gunzburger and Lili Ju and Zhu Wang},
  year={2021},
  eprint={2110.03442},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
