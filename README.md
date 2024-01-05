# Protein Structure Transformer

The repository implements the Protein Structure Transformer (PST). The PST performs a simple modification to the attention mechanism of a sequence-based protein-language model to add structural information. The PST is described in the paper [Protein Structure Transformer][1].


**TL;DR**: 

## Citation

Please use the following to cite our work:

```bibtex
```


## A short description of PST

### Overview of PST

The structure extractor adopts a uses a GNN on subgraphs of the main 8Å protein structure graph to extract structural information on the resiude level. The resulting residue-level representation can then be linearly projected to the $Q$, $K$ and $V$ vectors of the self-attention mechanism of any (pretrained) transformer model pretrained on larger corpuses of sequences. The pretraining dataset to learn the weights of the structure extractor can much smaller than the pretraining dataset of the sequence-based backbone. 

### Example of PST with SAT with ESM-2 as backbone

Below you can see a typical setup of PST with the [SAT][3]-based structure extractor and ESM-2 as a backbone. The ESM-2 model weights are frozen during the training of the structure extractor. The structure extractor is trained on a small dataset of 350K proteins with predicted structure. The resulting PST model can then be finetuned on a downstream task, e.g. [torchdrug][5] or [proteinshake][4] tasks. However, as we show in the paper, such a finetuning is not necessary to achieve state-of-the-art results on many tasks.

![Model_Arch](figures/PST_SAT_ESM.png)

### Running inference using PST

To see how you can get better representations from your data using PST, see `./scripts/pst_inference.py` for more details.

### Training PST

You can find the script where we train PST variants in `./train_pst.py`.

## Installation

The dependencies are managed by [mamba][2]

```

mamba create -n pst python=3.9 pytorch torchvision torchaudio pytorch-cuda=12.1 pyg lightning nvitop pytorch-scatter -c pytorch -c nvidia -c pyg
mamba activate pst
pip install proteinshake isort black fair-esm pyprojroot einops torch_geometric==2.3.1 pandas easydict pyprojroot scikit-learn flaml

```

Once you have activated the environment and installed all dependencies, run:

```bash
source s
```



[1]: https://arxiv.org/abs/TODO
[2]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
[3]: https://arxiv.org/abs/2202.03036
[4]: https://proteinshake.ai/
[5]: https://torchdrug.ai/ 