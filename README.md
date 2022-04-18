# SecFedNIDS
## Cite
```
Zhao Zhang, Yong Zhang, Da Guo, Lei Yao, Zhao Li, SecFedNIDS: Robust defense for poisoning attack against federated learning-based network intrusion detection system, Future Generation Computer Systems, 2022

https://www.sciencedirect.com/science/article/pii/S0167739X22001339
```
## (key) Requirements 
- Pytorch
- pyod

## Running Experiments
Launch the label-flipping attack against FL-based NIDS on UNSW-NB15 dataset 

Conduct the poisoned model detection:

```
python main_poisoned_model_det.py 
```
Conduct the Poisoned data detection:

```
python main_poisoned_data_det.py
```

