# Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders (KDD 2023 - [Link](https://dl.acm.org/doi/10.1145/3580305.3599444) - [Arxiv](https://arxiv.org/abs/2305.18612))

## Datasets

* The datasets being used in the paper can be found in this [link](https://drive.google.com/file/d/1kmY2MMlga1ryasGsAHXslKNI3F2l19IT/view?usp=share_link).

* After downloading and unzipping the datasets, please move them into the `dataset` folder under the root of this repo.


## Run Experiments

* `run_imputation.py` is used to compute the metrics for the deep imputation methods. An example of usage is

	```
	python run_imputation.py --config config/pogevon/air36.yaml --in-sample False
	```

* When running experiments for `PEMS-BA`, `PEMS-LA` and `PEMS-SD` datasets, one needs to change the `subdataset_name` value in config file `pems.ymal` to `'PEMS-04'`, `'PEMS-07'` and `'PEMS-11'` respectively.

## Requirements

We run all the experiments in `python 3.8`, see `requirements.txt` for the list of `pip` dependencies.

## Bibtex reference

If you find this code useful please consider to cite our paper:

```
@inproceedings{wang2023networked,
  title={Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders},
  author={Wang, Dingsu and Yan, Yuchen and Qiu, Ruizhong and Zhu, Yada and Guan, Kaiyu and Margenot, Andrew and Tong, Hanghang},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2256--2268},
  year={2023}
}
```

## Acknowledgement
This repo is based on the implementations of [GRIN](https://github.com/Graph-Machine-Learning-Group/grin) and thanks for their contribution.
