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

| Model name   | ESM-2 backbone        | #Layers | Embed dim | Notes             | Model URL                                                  |
| ------------ | --------------------- | ------- | --------- | ----------------- | ---------------------------------------------------------- |
| `pst_t6`     | `esm2_t6_8M_UR50D`    | 6       | 320       | Standard          | [link](https://datashare.biochem.mpg.de/s/ac9ufZ0NB2IrkZL) |
| `pst_t6_so`  | `esm2_t6_8M_UR50D`    | 6       | 320       | Train struct only | [link](https://datashare.biochem.mpg.de/s/ARzKycmMQePvLXs) |
| `pst_t12`    | `esm2_t12_35M_UR50D`  | 12      | 480       | Standard          | [link](https://datashare.biochem.mpg.de/s/fOSIwJAIKLYjFe3) |
| `pst_t12_so` | `esm2_t12_35M_UR50D`  | 12      | 480       | Train struct only | [link](https://datashare.biochem.mpg.de/s/qRvDPTfExZkq38f) |
| `pst_t30`    | `esm2_t30_150M_UR50D` | 30      | 640       | Standard          | [link](https://datashare.biochem.mpg.de/s/a3yugJJMe0I0oEL) |
| `pst_t30_so` | `esm2_t30_150M_UR50D` | 30      | 640       | Train struct only | [link](https://datashare.biochem.mpg.de/s/p73BABG81gZKElL) |
| `pst_t33`    | `esm2_t33_650M_UR50D` | 33      | 1280      | Standard          | [link](https://datashare.biochem.mpg.de/s/RpWYV4o4ka3gHvX) |
| `pst_t33_so` | `esm2_t33_650M_UR50D` | 33      | 1280      | Train struct only | [link](https://datashare.biochem.mpg.de/s/xGpS7sIG7k8DZX0) |

## Usage

### Installation

The dependencies are managed by [mamba][2]

```
mamba env create -f environment.yaml 
mamba activate pst
pip install -e .
```

Optionally, you can install the following dependencies to run the experiments:

```
pip install torchdrug
```

Finally, you can also directly install the dependencies from the `environment.yml` file:

```
mamba create -n pst python=3.9
mamba env update -n pst --file environment.yaml
pip install -e . 
```


### Quick start: extract representations of protein structures using PST

You can PST to simply extract representations of protein structures stored in PDB files. Just run

```bash
python scripts/pst_extract.py --help
```

If you want to work with your own dataset, just create a `my_dataset` directory in `scripts` and put all the PDB files into `my_dataset/raw/`,
and run:

```bash
python scripts/pst_extract.py --datadir ./scripts/my_dataset --model pst_t33_so --include_seq
```

### Use PST for protein function prediction

You can use PST to perform Gene Ontology prediction, Enzyme Commission Number prediction and any other protein function prediction tasks.

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

### Reproducibility datasets

We have folded structures that were not available in the PDB for our VEP datasets. You can download the dataset from [here](https://datashare.biochem.mpg.de/s/2UgA8kBwmCAVEsL), and unzip it in ./datasets, provided your current path is the root of this repository. Similarly, download the SCOP dataset [here](https://datashare.biochem.mpg.de/s/2yUwpK7pt2TMQ5E).


[1]: https://arxiv.org/abs/TODO
[2]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
[3]: https://arxiv.org/abs/2202.03036
[4]: https://proteinshake.ai/
[5]: https://torchdrug.ai/
[6]: https://github.com/facebookresearch/esm/