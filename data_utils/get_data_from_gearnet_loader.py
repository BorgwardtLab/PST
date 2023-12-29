import logging
import os
import pprint
import random
import sys
from pathlib import Path

import esm
import numpy as np
import protein_rep_learning.util as util
import torch
from protein_rep_learning.utils import (compute_angle_features, get_residue,
                                        write_dict_to_json, write_pickle)
from torch.utils import data
from torchdrug import core, datasets, models, tasks, utils  # noqa
from torchdrug.utils import comm
from tqdm.rich import tqdm

module = sys.modules[__name__]
logger = logging.getLogger(__name__)


def main():
    esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
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
    train_loader = data.DataLoader(train_set, 1, num_workers=4)
    valid_loader = data.DataLoader(valid_set, 1, num_workers=4)
    test_loader = data.DataLoader(test_set, 1, num_workers=4)
    general_dest = (
        "/fs/pool/pool-hartout/Documents/Git/protein_rep_learning/data/torchdrug_pkl/"
    )

    task_dest = (
        general_dest
        + cfg.dataset["class"]
        + "/"
        + cfg.dataset["path"].split("/")[-1]
        + "/"
    )
    Path(task_dest).mkdir(parents=True, exist_ok=True)

    alphabet = train_loader.dataset[0]["graph"].alphabet2id

    write_dict_to_json(
        task_dest + "/alphabet.json",
        alphabet,
    )
    write_dict_to_json(
        task_dest + "/targets.json",
        dataset.targets,
    )

    train_content = list()

    for batch in tqdm(train_loader.dataset):
        content = {
            "position": batch["graph"].node_position[batch["graph"].atom_name == 1],
            "residue": batch["graph"].data_dict["residue_type"],
            "targets": batch["targets"],
        }
        angle_features = compute_angle_features(content["position"])
        sequence = get_residue(
            content["residue"], angle_features, alphabet, esm_alphabet
        )
        content["sequence"] = sequence
        train_content.append(content)

    write_pickle(task_dest + "train.pkl", train_content)

    valid_content = list()
    for batch in tqdm(valid_loader.dataset):
        content = {
            "position": batch["graph"].node_position[batch["graph"].atom_name == 1],
            "residue": batch["graph"].data_dict["residue_type"],
            "targets": batch["targets"],
        }
        angle_features = compute_angle_features(content["position"])
        sequence = get_residue(
            content["residue"], angle_features, alphabet, esm_alphabet
        )
        content["sequence"] = sequence
        valid_content.append(content)

    write_pickle(task_dest + "valid.pkl", valid_content)

    test_content = list()
    for batch in tqdm(test_loader.dataset):
        content = {
            "position": batch["graph"].node_position[batch["graph"].atom_name == 1],
            "residue": batch["graph"].data_dict["residue_type"],
            "targets": batch["targets"],
        }
        angle_features = compute_angle_features(content["position"])
        sequence = get_residue(
            content["residue"], angle_features, alphabet, esm_alphabet
        )
        content["sequence"] = sequence
        test_content.append(content)
    write_pickle(task_dest + "test.pkl", test_content)


if __name__ == "__main__":
    main()
