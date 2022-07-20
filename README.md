# Conditional Similarity Network (CSN) on Farfetch Dresses 

This repository is a fork of the official [Conditional Similarity Network](https://github.com/andreasveit/conditional-similarity-networks) repository, 
implementing the model in the [paper](https://arxiv.org/abs/1603.07810) presented at CVPR 2017. 

I did some refactoring and modified the code to train on [Farfetch dresses](https://www.farfetch.com/hk/shopping/women/dresses-1/items.aspx).
The model can be used for explainable similar product recommendation. The reason to choosing Farfetch is that their 
products are well annotated with detailed properties like color, sleeves, collar, etc, as shown in the following image:

![](assets/farfetch_website.png)


## Data
1. Go to this [Google Drive](https://drive.google.com/file/d/1SYQrE3oX3_OtgMOWv9EikXva9h7OMaPn/view?usp=sharing) link and download the zip file to the project directory
2. Unzip the file

## Usage
The network can be simply trained with `python main.py` or with optional arguments for different hyperparameters:
```sh
$ python main.py --name {your experiment name} --learned --num_traintriplets 200000
```

Training progress can be easily tracked with [visdom](https://github.com/facebookresearch/visdom) using the `--visdom` flag. It keeps track of the learning rate, loss, training and validation accuracy both for all triplets as well as separated for each notion of similarity, the embedding norm, mask norm as well as the masks.

![](assets/visdom.png)

By default the training code keeps track of the model with the highest performance on the validation set. Thus, after the model has converged, it can be directly evaluated on the test set as follows
```sh
$ python main.py --test --resume runs/{your experiment name}/model_best.pth.tar
```


## Model Details
What makes images similar? To measure the similarity between images, they are typically embedded in a feature-vector space, in which their distance preserve the relative dissimilarity. However, when learning such similarity embeddings the simplifying assumption is commonly made that images are only compared to one unique measure of similarity.

[Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) address this shortcoming by learning a nonlinear embeddings that gracefully deals with multiple notions of similarity within a shared embedding. Different aspects of similarity are incorporated by assigning responsibility weights to each embedding dimension with respect to each aspect of similarity.

![](assets/csn_overview.png)

Images are passed through a convolutional network and projected into a nonlinear embedding such that different dimensions encode features for specific notions of similarity. Subsequent masks indicate which dimensions of the embedding are responsible for separate aspects of similarity. We can then compare objects according to various notions of similarity by selecting an appropriate masked subspace.
