#!/bin/sh

python test_inet_c.py /path/to/ImageNet-C /path/to/model/checkpoints
python test_inet_a.py /path/to/ImageNet-A /path/to/model/checkpoints
python test_inet_clean.py /path/to/ImageNet /path/to/model/checkpoints
python test_inet_r.py /path/to/ImageNet-R /path/to/model/checkpoints
python test_inet_sk.py /path/to/ImageNet-Sketch /path/to/model/checkpoints
python test_inet_v2.py /path/to/ImageNet-V2 /path/to/model/checkpoints
python test_onet.py /path/to/ObjectNet /path/to/model/checkpoints
python test_inet_all_candidates.py /path/to/ImageNet /path/to/model/checkpoints
python test_inet_p.py /path/to/ImageNet-P /path/to/model/checkpoints
python test_inet_syn2nat.py /path/to/ImageNet /path/to/model/checkpoints
