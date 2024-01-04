import argparse
import contextlib
import json
import logging
import math
import os
import pickle
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import easydict
import esm
import jinja2
import joblib
import torch
import yaml
from fastavro import reader as avro_reader
from jinja2 import meta
from joblib import Parallel, delayed
from sklearn.neighbors import radius_neighbors_graph
from torch import Tensor
from torch import distributed as dist
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_scatter import scatter
from torchdrug import core, utils
from torchdrug.utils import comm
from tqdm import tqdm

AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.task["class"],
        cfg.dataset["class"],
        cfg.task.model["class"],
        time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument(
        "-s", "--seed", help="random seed for PyTorch", type=int, default=1024
    )

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_downstream_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning(
            "#train: %d, #valid: %d, #test: %d"
            % (len(train_set), len(valid_set), len(test_set))
        )

    if cfg.task["class"] == "MultipleBinaryClassification":
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    else:
        cfg.task.task = dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {
                "params": solver.model.model.parameters(),
                "lr": cfg.optimizer.lr * cfg.lr_ratio,
            },
            {"params": solver.model.mlp.parameters(), "lr": cfg.optimizer.lr},
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.task.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {
                "params": solver.model.model.sequence_model.parameters(),
                "lr": cfg.optimizer.lr * cfg.sequence_model_lr_ratio,
            },
            {
                "params": solver.model.model.structure_model.parameters(),
                "lr": cfg.optimizer.lr,
            },
            {"params": solver.model.mlp.parameters(), "lr": cfg.optimizer.lr},
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
        task.model.load_state_dict(model_dict)

    return solver, scheduler


def build_pretrain_solver(cfg, dataset):
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#dataset: %d" % (len(dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    if "fix_sequence_model" in cfg:
        if cfg.task["class"] == "Unsupervised":
            model_dict = cfg.task.model.model
        else:
            model_dict = cfg.task.model
        assert model_dict["class"] == "FusionNetwork"
        for p in task.model.model.sequence_model.parameters():
            p.requires_grad = False
    cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, dataset, None, None, optimizer, **cfg.engine)

    return solver


def to_dense_batch(x, ptr, return_mask=False, fill_value=0.0):
    batch_size = len(ptr) - 1
    num_nodes = torch.diff(ptr)
    max_num_nodes = num_nodes.max().item()
    all_num_nodes = ptr[-1].item()

    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1

    device = x[0].device if isinstance(x, (list, tuple)) else x.device

    batch = torch.arange(batch_size, dtype=torch.long, device=device)
    batch = batch.repeat_interleave(num_nodes)
    idx = torch.arange(all_num_nodes, dtype=torch.long, device=device)
    idx = (idx - ptr[batch]) + batch * max_num_nodes

    size = [batch_size * max_num_nodes]
    if isinstance(x, (list, tuple)):
        new_x = []
        for j in range(len(x)):
            xj = x[j].new_full(size + list(x[j].size())[1:], fill_value)
            xj[idx] = x[j][:all_num_nodes]
            xj = xj.view([batch_size, max_num_nodes] + list(x[j].size())[1:])
            if cls_tokens:
                xj[:, -1, ...] = x[j][all_num_nodes:]
            new_x.append(xj)
    else:
        new_x = x.new_full(size + list(x.size())[1:], fill_value)
        new_x[idx] = x[:all_num_nodes]
        new_x = new_x.view([batch_size, max_num_nodes] + list(x.size())[1:])
        if cls_tokens:
            new_x[:, -1, ...] = x[all_num_nodes:]

    if return_mask:
        mask = torch.ones(batch_size * max_num_nodes, dtype=torch.bool, device=device)
        mask[idx] = 0
        mask = mask.view(batch_size, max_num_nodes)
        if cls_tokens:
            mask[:, -1] = 0
        return new_x, mask
    return new_x


def unpad_dense_batch(x, mask, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
    if cls_tokens:
        new_x = x[:, :-1][~mask[:, :-1]]
        new_x = torch.cat((new_x, x[:, -1]))
    else:
        new_x = x[~mask]
    return new_x


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Copied from PyG implementation"""
    if ptr is not None:
        offsets = ptr[1:] - ptr[:-1]
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    elif batch is not None:
        offsets = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there does not exist any (public) API to preven this :(
    return torch.nested.as_nested_tensor(xs)


def from_nested_tensor(
    x: Tensor,
    return_batch: bool = False,
):
    """Copied from PyG implementation"""
    if not x.is_nested:
        raise ValueError("Input tensor in 'from_nested_tensor' is not nested")

    sizes = x._nested_tensor_size()

    for dim, (a, b) in enumerate(zip(sizes[0, 1:], sizes.t()[1:])):
        if not torch.equal(a.expand_as(b), b):
            raise ValueError(
                f"Not all nested tensors have the same size "
                f"in dimension {dim + 1} "
                f"(expected size {a.item()} for all tensors)"
            )

    out = x.contiguous().values()
    out = out.view(-1, *sizes[0, 1:].tolist())

    if not return_batch:
        return out

    batch = torch.arange(x.size(0), device=x.device)
    batch = batch.repeat_interleave(sizes[:, 0].to(batch.device))

    return out, batch


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_inverse_sqrt_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        return math.sqrt(warmup_epochs / epoch)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def flatten_lists(lists: List) -> List:
    """Removes nested lists.

    Args:
        lists (List): List of lists to flatten

    Returns:
        List: Flattened list
    """
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    Code stolen from https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def distribute_function(
    func: Callable,
    X: Iterable,
    n_jobs: int,
    description: str = "",
    total: int = 1,
    **kwargs,
) -> Any:
    """Distributes function `func` over iterable `X` using `n_jobs` cores.

    Args:
        func (Callable): function to be distributed
        X (Iterable): iterable over which the function is distributed
        n_jobs (int): number of cores to use
        description (str, optional): Description of the progress. Defaults to "".
        total (int, optional): Total number of elements in `X`. Defaults to 1.

    Returns:
        Any: result of the `func` applied to `X`.
    """

    if total == 1:
        total = len(X)  # type: ignore

    with tqdm_joblib(tqdm(desc=description, total=total)):
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    return device


def make_batches(l, n):
    """Make chunks of size n from list l"""
    return [l[i : i + n] for i in range(0, len(l), n)]


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def write_dict_to_json(file_path: str, data: Dict) -> None:
    """Writes a dictionary to a JSON file.

    Args:
        file_path: The path where the JSON file will be stored.
        data: The dictionary to be written to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def read_json_to_dict(file_path: str) -> Dict:
    """Reads a JSON file into a dictionary.

    Args:
        file_path: The path of the JSON file to be read.

    Returns:
        A dictionary representation of the JSON file content.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_pickle(file_path: str, data: Any) -> None:
    """Writes data to a pickle file.

    Args:
        file_path (str): The path to the file to write to.
        data (Any): The data to pickle.

    Returns:
        None
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(file_path: str) -> Any:
    """Reads data from a pickle file.

    Args:
        file_path (str): The path to the file to read from.

    Returns:
        Any: The data read from the file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def find_keys_by_value(d: Dict[Any, Any], target_value: Any) -> List[Any]:
    """Find all keys in a dictionary that map to a given value.

    Args:
        d (Dict[Any, Any]): The dictionary to search.
        target_value (Any): The value to search for.

    Returns:
        List[Any]: A list of keys that map to the target value.
    """
    return [key for key, value in d.items() if value == target_value]


def get_rbf(X):
    D = torch.sqrt(torch.sum((X[:, None, :] - X[None, :, :]) ** 2, -1) + 1e-6)
    return rbf(D, 16)


def get_residue(residue, angle_features, alphabet, esm_alphabet):
    sequence_raw = "".join(
        [find_keys_by_value(alphabet, res)[0] for res in residue.tolist()]
    )

    # Remove trailing spaces
    sequence_str = [res for res in sequence_raw.strip().replace(" ", "X")]
    seq_tmp = list()
    for res in sequence_str:
        if "X" in res:
            seq_tmp.append(esm_alphabet.get_idx("<unk>"))
        else:
            seq_tmp.append(esm_alphabet.get_idx(res))
    sequence = torch.tensor(seq_tmp)
    # Add padding to match angle_features length
    padding_value = esm_alphabet.get_idx("<unk>")
    padding_length = angle_features.shape[0] - sequence.shape[0]
    if padding_length < 0:
        # Remove trailing residues
        sequence = sequence[:padding_length]
    else:
        padding = torch.full((padding_length,), padding_value, dtype=torch.long)
        sequence = torch.cat([sequence, padding])
    return sequence


def featurize_ca(sample, esm_alphabet=None, eps=8, use_rbfs=True, mask_cls_idx=True):
    if sample["position"].shape[0] == 0:
        return None
    else:
        graph_adj = radius_neighbors_graph(
            sample["position"], radius=eps, mode="connectivity"
        )
        if use_rbfs:
            rbf_dist = get_rbf(sample["position"]).squeeze()
            edge_attr = rbf_dist[row, col]
        else:
            edge_attr = None
        # mask out rbfs that are not connected
        row, col = torch.nonzero(torch.tensor(graph_adj.todense()), as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)

        # Map residue integeters to sequences using alphabet where keys are letters and integers are values.
        sequence = torch.cat(
            [
                torch.LongTensor([esm_alphabet.cls_idx]),
                sample["sequence"],
                torch.LongTensor([esm_alphabet.eos_idx]),
            ]
        )
        edge_index = edge_index + 1

        if mask_cls_idx:
            idx_mask = torch.ones((len(sequence),), dtype=torch.bool)
            idx_mask[0] = idx_mask[-1] = False
        else:
            idx_mask = None

        return Data(
            x=sequence,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=sample["targets"].unsqueeze(1).T,
            idx_mask=idx_mask,
        )


def read_avro_file(path):
    content = list()
    with open(
        path,
        "rb",
    ) as file:
        total = int(avro_reader(file).metadata["number_of_proteins"])

    with open(path, "rb") as file:
        for x in tqdm(avro_reader(file), desc="Reading avro file", total=total):
            content.append(x)
    return content


def get_graph_from_ps_protein_worker(
    protein, eps=8.0, use_rbfs=True, mask_cls_idx=False
):
    """Converts an avro dictionary to a pytorch geometric graph."""
    sequence = torch.LongTensor(
        [
            esm_alphabet.get_idx(res)
            for res in esm_alphabet.tokenize(protein["protein"]["sequence"])
        ]
    )

    coords = torch.tensor(
        [
            protein["residue"]["x"],
            protein["residue"]["y"],
            protein["residue"]["z"],
        ]
    ).T

    graph_adj = radius_neighbors_graph(coords, radius=eps, mode="connectivity")
    row, col = torch.nonzero(torch.tensor(graph_adj.todense()), as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)

    sequence = torch.cat(
        [
            torch.LongTensor([esm_alphabet.cls_idx]),
            sequence,
            torch.LongTensor([esm_alphabet.eos_idx]),
        ]
    )

    edge_index = edge_index + 1

    if use_rbfs:
        rbf_dist = get_rbf(coords).squeeze()
        edge_attr = rbf_dist[row, col]
    else:
        edge_attr = None

    return Data(edge_index=edge_index, x=sequence, edge_attr=edge_attr)


def get_graph_from_ps_protein(protein, eps=8.0, use_rbfs=True):
    if isinstance(protein, list):
        res = list()
        for p in protein:
            res.append(get_graph_from_ps_protein_worker(p, eps, use_rbfs))
        return res
    else:
        return get_graph_from_ps_protein_worker(protein, eps, use_rbfs)
