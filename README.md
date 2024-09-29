# BaseNet
BaseNet: A Transformer-Based Toolkit for Nanopore Sequencing Signal Decoding

# How to install
run the following commands:  
```
git clone https://github.com/liqingwen98/BaseNet.git
cd BaseNet
pip install -e .
```

# How to use
for training
```python
from basenet.models.joint_model import Model

model = Model()
loss = model(signals, signal_lengths, bases, base_lengths)
loss.backward()
```

for inference
```python
from basenet.models.joint_model import Model
from fast_ctc_decode import beam_search

beamsize=5
threshold=1e-3
alphabet = [ "N", "A", "C", "G", "T" ]

model = Model()
logits = model(signals).transpose(1,0)
for logit in torch.exp(logits).cpu().numpy():
    seq, path = beam_search(logit, alphabet, beamsize, threshold)
```

Because of the github store limitation, check point files can not be added to my repository. 

If you need check point files or have any question, welcome to contact me: li.qing.wen@foxmail.com

## Citations
``` bibtex
@article{LI20243430,
title = {BaseNet: A transformer-based toolkit for nanopore sequencing signal decoding},
journal = {Computational and Structural Biotechnology Journal},
year = {2024},
doi = {https://doi.org/10.1016/j.csbj.2024.09.016},
author = {Qingwen Li and Chen Sun and Daqian Wang and Jizhong Lou},
```
