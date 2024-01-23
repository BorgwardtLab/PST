# Protein Structure Transformer

The repository implements the Protein Structure Transformer (PST). The PST model extends the protein sequence model [ESM-2][6] to extract representations of protein structures. Full details of PST can be found in the [paper][1].

## Citation

Please use the following to cite our work:

```bibtex
```


## Overview of PST

PST uses a structure extractor to incorporate protein structures into existing protein language models (PLMs) such as [ESM-2][6].
The structure extractor adopts a GNN to extract subgraph representations of the 8Å-neighborhood protein structure graph at each residue (i.e., nodes on the graph). The resulting residue-level subgraph representations are then add to the $Q$, $K$ and $V$ matrices of **each** self-attention block of any (pretrained) transformer model (here we use **ESM-2**) pretrained on larger corpuses of sequences. We name the resulting model PST, which can be trained on any protein structure dataset, by either updating the full model weights or only the weights in the structure extractor. The pretraining dataset could be much smaller than the pretraining dataset of the base sequence model, e.g., SwissProt with only 550k protein structures. 

Below you can find an overview of PST with ESM-2 as the sequence backbone. The ESM-2 model weights are frozen during the training of the structure extractor. The structure extractor was trained on AlphaFold SwissProt, a dataset of 550K proteins with predicted structures. The resulting PST model can then be finetuned on a downstream task, e.g., [torchdrug][5] or [proteinshake][4] tasks. PST can also be used to extract representations of protein structures.

![Overview of PST](assets/overview.png)

## Pretrained models

| Model name | #Layers | Embed dim | Train struct only | Model URL |
|:-----------|:-------:|:---------:|:-----------------:|:---------:|
| pst_t6     |    6    |    320    |       False       |   link    |
| pst_t6_so  |    6    |    320    |       True        |   link    |


## Usage

### Installation

The dependencies are managed by [mamba][2]

```

mamba create -n pst python=3.9 pytorch torchvision pytorch-cuda=12.1 pyg lightning nvitop pytorch-scatter -c pytorch -c nvidia -c pyg
mamba activate pst
pip install proteinshake isort black fair-esm pyprojroot einops torch_geometric==2.3.1 pandas easydict pyprojroot scikit-learn flaml

```

### Quick start: extract protein representations using PST

To see how you can get better representations from your data using PST, see `./scripts/pst_inference.py` for more details.

### Use PST for protein function prediction

You can use PST to perform Gene Ontology prediction, Enzyme Commission prediction and any protein function prediction tasks.

#### Fixed representations

To train an MLP on top of the representations extracted by the pretrained PST models for Enzyme Commission prediction, run:

```bash
python experiments/fixed/predict_gearnet.py dataset=gearnet_ec # dataset=gearnet_go_bp, gearnet_go_cc or gearnet_go_mf for GO prediction
```

#### Finetune PST

To finetune the PST model for function prediction tasks, run:

```bash
python experiments/finetune/finetune_gearnet.py dataset=gearnet_ec # dataset=gearnet_go_bp, gearnet_go_cc or gearnet_go_mf for GO prediction
```

### Pretrain PST on AlphaFold Swissprot

Run the following code to train a PST model based on the 6-layer ESM-2 model by only training the structure extractor:

```bash
python train_pst.py base_model=esm2_t6 model.train_struct_only=true
```

You can replace `esm2_t6` with `esm2_t12`, `esm2_t30`, `esm2_t33` or any pretrained ESM-2 model.



[1]: https://arxiv.org/abs/TODO
[2]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
[3]: https://arxiv.org/abs/2202.03036
[4]: https://proteinshake.ai/
[5]: https://torchdrug.ai/
[6]: https://github.com/facebookresearch/esm/