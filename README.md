# SecFedNIDS

## Cite 
Zhao Z, Yong Z, Da G, Lei Y, Zhao L,SecFedNIDS: Robust defense for poisoning attack against federated learning-based network intrusion detection system,
Future Generation Computer Systems,Volume 134,2022,Pages 154-169,https://doi.org/10.1016/j.future.2022.04.010.

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

