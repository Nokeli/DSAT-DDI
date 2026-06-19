# DSAT-DDI

**Dual-channel Structure-Aware Transformer for Drug–Drug Interaction (DDI) prediction, with a focus on the cold-start (inductive) setting.**

DSAT predicts whether two drugs interact by jointly modeling, for each drug, two complementary views:

- a **molecular-graph channel** — a Structure-Aware Transformer (SAT) over the drug's atom graph (intrinsic, available for any drug);
- a **knowledge-graph channel** — a SAT over the drug's *k*-hop subgraph in a biomedical knowledge graph (relational context).

The two channel representations are fused (cross-attention) and scored with an MLP.

This repository builds on the [Structure-Aware Transformer (SAT, ICML 2022)](https://arxiv.org/abs/2202.03036) backbone and on the dual-channel DDI formulation of [TIGER (AAAI 2024)](https://github.com/Blair1213/TIGER).

## What's in this method

On top of the dual-channel SAT backbone, DSAT adds four components aimed at **cold-start generalization** (predicting interactions for drugs unseen during training):

1. **InfoMax self-supervision** — mutual-information maximization between each channel's pooled representation and its substructure embeddings, grounding representations in transferable structure rather than memorized identity.
2. **Degree encoding** — an inductive structural signal added to node features (valid for unseen drugs).
3. **Molecule-bridged inductive KG center** — the KG subgraph's center (drug) node is given an embedding derived from the drug's molecular representation instead of a purely transductive ID lookup, plus **cold-start simulation** (identity dropout during training) so the molecular pathway stays predictive when a drug's identity is unknown.
4. **Relation-aware attention bias** — edge relation types are injected as a bias into the KG channel's self-attention.

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
- split folders, e.g. `data/cold_one_strict/fold_{0..4}/{train,valid,test}.npy`, each an `N×3` array of `[drug1, drug2, label]`.

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

## Results (5-fold, strict cold-start)

| Setting | Model | AUROC | AUPR |
|---|---|---|---|
| `cold_one_strict` | base (dual-channel SAT) | 0.6746 ± 0.0111 | 0.6307 ± 0.0144 |
| `cold_one_strict` | **DSAT (full)** | **0.7010 ± 0.0117** | **0.6673 ± 0.0097** |
| `cold_both_strict` | DSAT (full) | 0.5925 ± 0.0320 | 0.5885 ± 0.0264 |

## Acknowledgements

This work builds on [SAT](https://github.com/BorgwardtLab/SAT) (Chen et al., ICML 2022) and [TIGER](https://github.com/Blair1213/TIGER) (Su et al., AAAI 2024). Please cite those works if you use the corresponding components.

## License

See [LICENSE](LICENSE).
