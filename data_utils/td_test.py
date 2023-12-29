import json
import logging
import os
import pickle
import pprint
import random
import sys
from typing import Any, Dict

import numpy as np
import protein_rep_learning.util as util
import torch
from torch.utils import data
from torchdrug import core, datasets, models, tasks, utils  # noqa
from torchdrug.utils import comm
from tqdm.rich import tqdm

module = sys.modules[__name__]
logger = logging.getLogger(__name__)


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


def main():
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    cfg = util.load_config(args.config, context=vars)
    util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = dataset.split()
    train_loader = data.DataLoader(train_set, 32, num_workers=4)
    valid_loader = data.DataLoader(valid_set, 32, num_workers=4)
    test_loader = data.DataLoader(test_set, 32, num_workers=4)

    alphabet = train_loader.dataset[0]["graph"].alphabet2id
    write_dict_to_json(
        "/fs/pool/pool-hartout/Documents/Git/protein_rep_learning/data/torchdrug_pkl/alphabet.json",
        alphabet,
    )

    train_content = list()
    dest = (
        "/fs/pool/pool-hartout/Documents/Git/protein_rep_learning/data/torchdrug_pkl/"
    )
    for batch in tqdm(train_loader.dataset):
        train_content.append(
            {
                "position": batch["graph"].data_dict["node_position"],
                "residue": batch["graph"].data_dict["residue_type"],
            }
        )
    write_pickle(dest + "train.pkl", train_content)

    valid_content = list()
    for batch in tqdm(valid_loader.dataset):
        valid_content.append(
            {
                "position": batch["graph"].data_dict["node_position"],
                "residue": batch["graph"].data_dict["residue_type"],
            }
        )
    write_pickle(dest + "valid.pkl", valid_content)

    test_content = list()
    for batch in tqdm(test_loader.dataset):
        test_content.append(
            {
                "position": batch["graph"].data_dict["node_position"],
                "residue": batch["graph"].data_dict["residue_type"],
            }
        )
    write_pickle(dest + "test.pkl", test_content)


if __name__ == "__main__":
    main()
