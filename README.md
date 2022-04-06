# A  Comparison of Neural Network Architectures for Reduced-Order Modeling

This is an implementation of the algorithms from:

A. Gruber, M. Gunzburger, L. Ju, and Z. Wang. *A Comparison of Neural Network Architectures for Data-Driven Reduced-Order Modeling*, (under review).  Preprint [available here.](https://arxiv.org/abs/2110.03442#),

which compare various ROM architectures for scientific applications.  In particular, we implement nonlinear-ROM architectures based on GCN2 graph convolutional (_GCN2), standard convolutional (_CNN), and fully connected (_FC) architectures.  Moreover, we also implement linear POD-ROMs which solve the 1-D Burger's equation and the 2-D Navier Stokes equations.

The *src* folder contains the source code for the nonlinear ROMs, and the examples from the paper are reproduced by running the files in the relevant *scripts* folder.  Network parameters can be changed at network initialization (see *src/networks*), and training parameters can be changed using the helper function "train_network" (see *src/utils*).  The POD-ROMs can be run from the functions in the iPython notebooks (these can also be run straight down).  


## Installation
The code was written and tested using Python 3.8 on Mac OSX 11.2.3.  The required dependencies are
* [Numpy](https://numpy.org/)
* [Pytorch](https://pytorch.org/)
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
* [Matplotlib](https://matplotlib.org/) (for visualization)
* [Scipy](https://scipy.org) (for NSE POD-ROM)
* [Netgen/Ngsolve](https://ngsolve.org) (for NSE POD-ROM)

## Data
The necessary heat equation and Burger's equation data for reproducing the experiments in the paper is found in the "datasets" folder.  Note that the larger file is stored using Git LFS due to its size (~500 MB).  The NSE data can be generated from the function "generate_NSE_data" found in the NSE_ROM iPython notebook.

To use your own dataset with our algorithms, load it into a data object with the following attributes:
- *xTrain, xValid* tensors representing training and validation solution snapshots.
- *pTrain, pValid* tensors representing training and validation points in parameter space, ordered consistently with the solution snapshots.
- *edge_index* (2 x nEdges) LongTensor encoding the sparse adjacency matrix (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)).
Take a look at the network classes in *src/networks* and the function "train_network" in *src/utils* for a better idea of what is used.

## Citation
Please cite [our paper](https://arxiv.org/pdf/2110.03442.pdf) if you use this code in your own work:
```
@article{gruber2022comparison,
  title={A comparison of neural network architectures for data-driven reduced-order modeling},
  author={Gruber, Anthony and Gunzburger, Max and Ju, Lili and Wang, Zhu},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={393},
  pages={114764},
  year={2022},
  publisher={Elsevier}
}