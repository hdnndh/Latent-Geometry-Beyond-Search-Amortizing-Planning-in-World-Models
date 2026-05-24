<h1 align="center">Latent Geometry Beyond Search:<br>Amortizing Planning in World Models</h1>

<p align="center">
<a href="https://arxiv.org/abs/2605.08732"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2605.08732-b31b1b.svg"/></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"/></a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2605.08732"><b>Paper</b></a> · <a href="demo.ipynb"><b>Ipynb Demo</b></a>
</p>

<p align="center"><i>
Hoang Nguyen*, Xiaohao Xu*†, Xiaonan Huang<br>
University of Michigan
</i></p>

---

Vision-based world models like [LeWM](https://github.com/lucas-maes/le-wm) learn compact latent spaces where planning should be easy, yet test-time control still requires thousands of sampling rollouts per decision. We show that when the latent geometry is sufficiently smooth, this search is unnecessary: a 1.5M-parameter MLP trained on frozen LeWM embeddings can map (current state, goal state, remaining horizon) directly to the next action in a single forward pass. Across Two-Room, Push-T, OGBench-Cube, and Reacher, this Goal-Conditioned Inverse Dynamics Model (GC-IDM) matches or exceeds CEM, MPPI, iCEM, and gradient-based planners while running 100-130x faster per decision.

<div align="center">

| Environment | GC-IDM | CEM | Speedup |
|:---:|:---:|:---:|:---:|
| Two-Room | **100.0%** | 84.0% | 104x |
| Push-T | **84.2%** | 82.5% | 106x |
| OGBench-Cube | **98.7%** | 67.0% | 130x |
| Reacher | **99.7%** | 70.3% | 110x |

</div>

A broader sweep over test-time planners confirms that GC-IDM's advantage is not specific to CEM:

<div align="center">

| Environment | GC-IDM | CEM | MPPI | iCEM | Gradient |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Two-Room | **100.0%** | 84.0% | 65.7% | 87.5% | 30.3% |
| Push-T | **84.2%** | 82.5% | 61.3% | 80.0% | 2.5% |
| OGBench-Cube | **98.7%** | 67.0% | 49.3% | 70.5% | 32.0% |
| Reacher | **99.7%** | 70.3% | 45.7% | 69.8% | 6.5% |

</div>

### Real-time (wall-clock)

<p align="center">
<img width="380" src="https://github.com/user-attachments/assets/03fb98ca-ab15-4f51-9350-2b6df451c819" />  <img width="380" src="https://github.com/user-attachments/assets/6bec0bb1-f4d8-4db4-b42f-59533063318e" /> 
</p><p align="center"> 
<img width="380" src="https://github.com/user-attachments/assets/41b7202a-47e3-49e0-841a-fa92ce1abfa6" />   <img width="380" src="https://github.com/user-attachments/assets/f5b9cd9c-8878-44e4-9d43-a7e9a3d91de2" />  
</p>

Both methods run on the same GPU, same episode, same start and goal. Each frame of the GIF corresponds to a fixed interval of wall-clock time.

### Environment steps

<p align="center">
<img width="380" src="https://github.com/user-attachments/assets/bbbdacf0-da19-4c7c-810f-53c739c37590" /> <img width="380" src="https://github.com/user-attachments/assets/10864428-8275-4bab-b452-2416d2f6ff5f" /> 
</p><p align="center">     
<img width="380" src="https://github.com/user-attachments/assets/7a7f2277-1b47-4391-89ad-5290021a208f" /> <img width="380" src="https://github.com/user-attachments/assets/ea94733e-c176-44f2-84c7-0a98525e2028" />
</p>

Both methods take the same number of environment steps. This isolates goal-reaching ability from compute cost.

## Resource requirements

Training is lightweight since it runs on pre-extracted embeddings, not images.

- **GPU RAM**: ~1 GB
- **System RAM**: ~4 GB
- **Training time**: ~20 min per environment on one GPU (after embedding extraction)
- **Embedding extraction** (one-time): ~4 GB GPU RAM 
do change the batch size in case your GPU have higher capacity.

## Setup

```bash
pip install torch torchvision
pip install "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git"
pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git"
pip install h5py hdf5plugin scikit-learn mujoco dm-control
export MUJOCO_GL=egl
```

Datasets and pretrained LeWM checkpoints: [HuggingFace collection](https://huggingface.co/collections/quentinll/lewm).

## Quick start

Open `demo.ipynb`. Set `ENV`, run all cells.
As render seems to change base on GPU, the h5 used in the experiments on L4 colab are shared in Google Drive links in the demo.

## Reproduce main results

```bash
# 1. Extract embeddings
python train_idm.py extract \
    --checkpoint $STABLEWM_HOME/checkpoints/tworoom/lewm \
    --h5 $STABLEWM_HOME/datasets/tworoom.h5 \
    --output ./embeddings/tworoom.npz

# 2. Train GC-IDM
python train_idm.py train \
    --embeddings ./embeddings/tworoom.npz \
    --output ./checkpoints/tworoom_gcidm.pt \
    --action-dim 2 --frameskip 1 \
    --epochs 50 --lr 1e-3 --batch-size 1024

# 3. Evaluate GC-IDM vs CEM
python eval_idm.py --dataset tworoom \
    --idm ./checkpoints/tworoom_gcidm.pt \
    --compare --num-eval 200
```

Repeat for `pusht`, `cube` (action-dim 5), `reacher`.

## Other solvers

```bash
python eval_othersolvers.py --solver mppi --dataset tworoom --num-eval 200
python eval_othersolvers.py --solver icem --dataset tworoom --num-eval 200
python eval_othersolvers.py --solver gradient --dataset tworoom --num-eval 200
```

## Project structure

```
gc-idm/
├── idm/
│   ├── model.py          # GoalConditionedIDM, IDMConfig
│   └── dataset.py        # Embedding extraction + dataset
├── train_idm.py          # Extract embeddings, train IDM
├── eval_idm.py           # GC-IDM vs CEM evaluation
├── eval_othersolvers.py  # MPPI, iCEM, Gradient comparison
└── demo.ipynb            # Python notebook
```

## Built on

This codebase builds on [LeWorldModel](https://github.com/lucas-maes/le-wm) for the pretrained JEPA world model and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation.


## Citation

```bibtex
@article{nguyen2026gcidm,
  title={Latent Geometry Beyond Search: Amortizing Planning in World Models},
  author={Nguyen, Hoang and Xu, Xiaohao and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2605.08732},
  year={2026}
}
```
