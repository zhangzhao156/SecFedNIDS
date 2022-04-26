# SecFedNIDS

## Cite 
@article{ZHANG2022154,
title = {SecFedNIDS: Robust defense for poisoning attack against federated learning-based network intrusion detection system},
journal = {Future Generation Computer Systems},
volume = {134},
pages = {154-169},
year = {2022},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2022.04.010},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X22001339},
author = {Zhao Zhang and Yong Zhang and Da Guo and Lei Yao and Zhao Li},
}

https://authors.elsevier.com/c/1ezXj,3q5xgP6x

## Requirements 
- Pytorch
- pyod
- ...

## Running Experiments
Attack model: Launch the label-flipping attack against FL-based NIDS on UNSW-NB15 dataset 

Conduct the poisoned model detection:

```
python main_poisoned_model_det.py 
```
Conduct the Poisoned data detection:

```
python main_poisoned_data_det.py
```

