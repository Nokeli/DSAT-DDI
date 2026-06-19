# DSAT-DDI

**Dual-channel Structure-Aware Transformer for Drug–Drug Interaction (DDI) prediction, with a focus on the cold-start (inductive) setting.**

DSAT predicts whether two drugs interact by jointly modeling, for each drug, two complementary views:

- a **molecular-graph channel** — a Structure-Aware Transformer (SAT) over the drug's atom graph;
- a **knowledge-graph channel** — a SAT over the drug's *k*-hop subgraph in a biomedical knowledge graph (relational context).

The two channel representations are fused (cross-attention) and scored with an MLP.

This repository builds on the [Structure-Aware Transformer (SAT, ICML 2022)](https://arxiv.org/abs/2202.03036) backbone and on the dual-channel DDI formulation of [TIGER (AAAI 2024)](https://github.com/Blair1213/TIGER).

## Installation

Tested with Python 3.9–3.11 and PyTorch + PyTorch Geometric. See `requirements.txt`.

```bash
conda create -n dsat python=3.9
conda activate dsat
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `torch-scatter`, `rdkit`, `numpy`, `scipy`, `scikit-learn`, `einops`, `tqdm`.

## Data

Data files are **not** included in this repository. The code expects, under `data/`:

- `new_smiles.txt` — one SMILES string per drug;
- `new_kg.txt` — biomedical knowledge-graph triples (`head tail relation`, space-separated);
- `new_ddi.txt` — DDI pairs with labels (`drug1 drug2 label`);

**Cold-start splits.** Evaluation uses *strict* drug-level splits where the train / validation / test drug sets are **mutually exclusive**:

- `cold_one_strict` — at least one drug in each test pair is unseen during training;
- `cold_both_strict` — both drugs in each test pair are unseen.

The first run builds and caches per-drug subgraphs and positional encodings as `*.json` files in `sat/` (these caches are git-ignored).

## Usage

```bash
cd sat
python train.py --fold 0          # train + evaluate on one fold
```

The data split read by `train.py` is set near the top of the training script (e.g. `data/cold_one_strict/fold_{}`). Run folds `0..4` and average for mean±std.

## Acknowledgements

This work builds on [SAT](https://github.com/BorgwardtLab/SAT) (Chen et al., ICML 2022) and [TIGER](https://github.com/Blair1213/TIGER) (Su et al., AAAI 2024). Please cite those works if you use the corresponding components.

