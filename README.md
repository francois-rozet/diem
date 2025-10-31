# Learning Diffusion Priors from Observations by Expectation Maximization

This repository contains the official implementation of the paper [Learning Diffusion Priors from Observations by Expectation Maximization](https://arxiv.org/abs/2405.13712) by [François Rozet](https://github.com/francois-rozet), [Gérôme Andry](https://github.com/gerome-andry), [François Lanusse](https://github.com/EiffL) and [Gilles Louppe](https://github.com/glouppe).

#### Abstract

Diffusion models recently proved to be remarkable priors for Bayesian inverse problems. However, training these models typically requires access to large amounts of clean data, which could prove difficult in some settings. In this work, we present DiEM, a novel method based on the expectation-maximization algorithm for training diffusion models from incomplete and noisy observations only. Unlike previous works, DiEM leads to proper diffusion models, which is crucial for downstream tasks. As part of our methods, we propose and motivate an improved posterior sampling scheme for unconditional diffusion models. We present empirical evidence supporting the effectiveness of our approach.

## Code

The majority of the code is written in [Python](https://www.python.org). Neural networks are built and trained using the [JAX](https://github.com/google/jax) automatic differentiation framework and the [Inox](https://github.com/francois-rozet/inox) library. We also rely on [torch-fidelity](https://github.com/toshas/torch-fidelity) to compute image quality metrics and [POT](https://github.com/PythonOT/POT) to compute Sinkhorn divergences. All dependencies are provided as a [conda](https://conda.io) environment file.

```
conda env create -f environment.yml
conda activate diem
```

To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com/overview.html) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [diem](diem) module as a package.

```
pip install -e .
```

### Organization

The [diem](diem) directory contains the implementations of the [neural networks](diem/nn.py), the [diffusion models](diem/diffusion.py) and various [helpers](diem/common.py).

The [manifold](experiments/manifold), [cifar](experiments/cifar) and [fastmri](experiments/fastmri) directories contain the scripts for the experiments (data generation, training and evaluation) as well as the notebooks that produced the figures of the paper. For example,

```
~/experiments/cifar $ python data.py
```

launches the CIFAR-10 data processing/generation jobs and

```
~/experiments/cifar $ python train.py
```

launches the training jobs with the default config (specified within the file).

For the accelerated MRI experiment, the `knee_singlecoil_{train|val}.tar.xz` archives should be obtained from the [NYU fastMRI Initiative database](https://fastmri.med.nyu.edu/).

The code for the evaluation of MMPS (in Appendix E) is provided in the separate [mmps-benchmark](https://github.com/francois-rozet/mmps-benchmark) repository.

## Citation

If you find this project useful for your research, please consider citing

```bib
@inproceedings{rozet2024learning,
  title={Learning Diffusion Priors from Observations by Expectation Maximization},
  author={Fran{\c{c}}ois Rozet and G{\'e}r{\^o}me Andry and Fran{\c{c}}ois Lanusse and Gilles Louppe},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=7v88Fh6iSM},
}
```

## Acknowledgements

#### Compute

The present research benefited from computational resources made available on Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region under the grant n°1910247. The computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the Fonds de la Recherche Scientifique de Belgique (F.R.S.-FNRS) under the grant n°2.5020.11 and by the Walloon Region.

#### NYU fastMRI

MRI data used in the preparation of this article were obtained from the NYU fastMRI Initiative database. As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at https://fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images.

```bib
@misc{Zbontar2018fastMRI,
  author = {Zbontar, Jure and Knoll, Florian and Sriram, Anuroop and Murrell, Tullie and Huang, Zhengnan and Muckley, Matthew J. and Defazio, Aaron and Stern, Ruben and Johnson, Patricia and Bruno, Mary and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzal, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Yakubova, Nafissa and Pinkerton, James and Wang, Duo and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.},
  title  = {fastMRI: An Open Dataset and Benchmarks for Accelerated MRI},
  year   = {2018},
  url    = {http://arxiv.org/abs/1811.08839},
}

@article{Knoll2020fastMRI,
  author  = {Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J. and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzalv, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Pinkerton, James and Wang, Duo and Yakubova, Nafissa and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.},
  journal = {Radiology: Artificial Intelligence},
  title   = {fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning},
  year    = {2020},
  doi     = {10.1148/ryai.2020190007},
}
```
