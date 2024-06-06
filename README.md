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

If you have any question, welcome to contact me: li.qing.wen@foxmail.com

## Citations
``` bibtex
@article {Li2024.06.02.597014,
	author = {Li, Qingwen and Sun, Chen and Wang, Daqian and Lou, Jizhong},
	title = {BaseNet: A Transformer-Based Toolkit for Nanopore Sequencing Signal Decoding},
	elocation-id = {2024.06.02.597014},
	year = {2024},
	doi = {10.1101/2024.06.02.597014},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/03/2024.06.02.597014},
	eprint = {https://www.biorxiv.org/content/early/2024/06/03/2024.06.02.597014.full.pdf},
	journal = {bioRxiv}
}
```
