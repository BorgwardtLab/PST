import copy
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from biopandas.pdb import PandasPdb
from proteinshake.datasets import Dataset
from proteinshake.utils import (download_url, extract_tar, load, save,
                                write_avro)
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
}


def tmalign_wrapper(pdb1, pdb2, return_superposition=False):
    """Compute TM score with TMalign between two PDB structures.
    Parameters
    ----------
    pdb1: str
        Path to PDB.
    arg2 : str
        Path to PDB.
    return_superposition: bool
        If True, returns a protein dataframe with superposed structures.
    Returns
    -------
    float
        TM score from `pdb1` to `pdb2`
    float
        TM score from `pdb2` to `pdb1`
    float
        RMSD between structures
    biopandas.pdb.pandas_pdb.PandasPdb
        If `return_superposition` contains coordinates of aligned structures.
    """
    assert (
        shutil.which("TMalign") is not None
    ), "No TMalign installation found. Go here to install : https://zhanggroup.org/TM-align/TMalign.cpp"
    try:
        if return_superposition:
            with tempfile.TemporaryDirectory() as tmpdir:
                out = subprocess.run(
                    [
                        "TMalign",
                        "-outfmt",
                        "2",
                        pdb1,
                        pdb2,
                        "-o",
                        os.path.join(tmpdir, "sup.pdb"),
                    ],
                    stdout=subprocess.PIPE,
                ).stdout.decode()
                df = PandasPdb().read_pdb(os.path.join(tmpdir, "sup.pdb"))
        else:
            out = subprocess.run(
                ["TMalign", "-outfmt", "2", pdb1, pdb2], stdout=subprocess.PIPE
            ).stdout.decode()

        path1, path2, TM1, TM2, RMSD, ID1, ID2, IDali, L1, L2, Lali = out.split("\n")[
            1
        ].split("\t")
    except Exception as e:
        print(e)
        return -1.0
    if return_superposition:
        return float(TM1), float(TM2), float(RMSD), df
    else:
        return float(TM1), float(TM2), float(RMSD)


class MutationDataset(Dataset):
    exlude_args_from_signature = ["id"]

    def __init__(self, **kwargs):
        self.id = self.available_ids()
        kwargs["use_precomputed"] = False
        kwargs["root"] = str(kwargs["root"])
        super().__init__(**kwargs)
        self.mutations = [pd.read_csv(f"{self.root}/raw/{id}.csv") for id in self.id]

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    def get_id_from_filename(self, filename):
        return filename.rstrip(".pdb")

    def download(self):
        """Implement me!
        This function downloads the raw data and preprocesses it in the following format, one for each protein:
        - {self.root}/raw/{self.id}.pdb # the reference pdb structure file
        - {self.root}/raw/{self.id}.csv # a table with columns 'mutations' (space separated), and 'y' (the measurements)
        """
        raise NotImplementedError


class DeepSequenceDataset(MutationDataset):
    meta_data = {
        "BF520_env_Bloom2018": "BF520",
        "BG505_env_Bloom2018": "BF505",
        "HG_FLU_Bloom2016": "P03454",
        "PA_FLU_Sun2015": "P15659",
        "POL_HV1N5-CA_Ndungu2014": "P12497",
        "BLAT_ECOLX_Ostermeier2014": "P62593",
        "BLAT_ECOLX_Ranganathan2015": "P62593",
        "BLAT_ECOLX_Tenaillon2013": "P62593",
        "BLAT_ECOLX_Palzkill2012": "P62593",
        "DLG4_RAT_Ranganathan2012": "P31016",
        "GAL4_YEAST_Shendure2015": "P04386",
        "HSP82_YEAST_Bolon2016": "P02829",
        "KKA2_KLEPN_Mikkelsen2014": "P00552",
        "MTH3_HAEAESTABILIZED_Tawfik2015": "P20589",
        "UBE4B_MOUSE_Klevit2013-singles": "Q9ES00",
        "YAP1_HUMAN_Fields2012-singles": "P46937",
        #'parEparD_Laub2015_all': 'F7YBW8',
        "AMIE_PSEAE_Whitehead": "P11436",
        "P84126_THETH_b0": "P84126",
        "TIM_SULSO_b0": "Q06121",
        "TIM_THEMA_b0": "Q56319",
        "IF1_ECOLI_Kishony": "P69222",
        "MK01_HUMAN_Johannessen": "P28482",
        "RASH_HUMAN_Kuriyan": "P01112",
        "RL401_YEAST_Fraser2016": "P0CH09",
        "BRCA1_HUMAN_RING": "P38398",
        "BRCA1_HUMAN_BRCT": "P38398",
        "B3VI55_LIPST_Whitehead2015": "B3VI55",
        "CALM1_HUMAN_Roth2017": "P0DP23",
        "TPK1_HUMAN_Roth2017": "Q9H3S4",
        "SUMO1_HUMAN_Roth2017": "P63165",
        "PABP_YEAST_Fields2013-singles": "P04147",
        "PABP_YEAST_Fields2013-doubles": "P04147",
        "RL401_YEAST_Bolon2013": "P0CH09",
        "RL401_YEAST_Bolon2014": "P0CH09",
        "TPMT_HUMAN_Fowler2018": "P51580",
        "PTEN_HUMAN_Fowler2018": "P60484",
        "HIS7_YEAST_Kondrashov2017": "P06633",
        "UBC9_HUMAN_Roth2017": "B0QYN7",
    }

    @classmethod
    def available_ids(cls):
        return list(cls.meta_data.keys())

    def get_raw_files(self):
        return [f"{self.root}/raw/{id}.pdb" for id in self.id]

    def download(self):
        url = "https://dl.fbaipublicfiles.com/fair-esm/examples/variant-prediction/data/raw_df.csv"
        download_url(url, f"{self.root}/raw/measurements.csv")
        measurements = pd.read_csv(f"{self.root}/raw/measurements.csv")

        for dataset, uniprot_id in tqdm(self.meta_data.items(), desc="Processing"):
            df = measurements[measurements["protein_name"] == dataset]
            df = df.rename(columns={"gt": "y", "mutant": "mutations"})
            df["mutations"] = df["mutations"].map(lambda x: " ".join(x.split(":")))
            df = df[["mutations", "y"]]
            df.to_csv(f"{self.root}/raw/{dataset}.csv", index=False)
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            root_af2 = self.root.split("/")[0] + "/" + "non_af2_db"
            root_af2 = f"{self.root}/non_af2_db"
            if not Path(f"{root_af2}/{uniprot_id}.pdb").exists():
                try:
                    download_url(url, f"{self.root}/raw/{dataset}.pdb")
                except requests.exceptions.HTTPError:
                    print(f"Could not download {url}")

            else:
                Path(f"{root_af2}/{uniprot_id}.pdb").rename(
                    Path(f"{self.root}/raw/{dataset}.pdb")
                )


class EnvisionDataset(MutationDataset):
    def download(self):
        pass


class SkempiDataset(MutationDataset):
    @classmethod
    def available_ids(self):
        return [
            "1A22",
            "1A4Y",
            "1ACB",
            "1AHW",
            "1AK4",
            "1BRS",
            "1CBW",
            "1CHO",
            "1CSE",
            "1DAN",
            "1DQJ",
            "1DVF",
            "1E96",
            "1EAW",
            "1EMV",
            "1F47",
            "1FC2",
            "1FCC",
            "1FFW",
            "1GC1",
            "1GCQ",
            "1H9D",
            "1HE8",
            "1IAR",
            "1JCK",
            "1JRH",
            "1JTG",
            "1KTZ",
            "1LFD",
            "1MAH",
            "1MQ8",
            "1NMB",
            "1PPF",
            "1R0R",
            "1REW",
            "1S1Q",
            "1TM1",
            "1UUZ",
            "1VFB",
            "1XD3",
            "1Z7X",
            "2FTL",
            "2G2U",
            "2I9B",
            "2J0T",
            "2JEL",
            "2O3B",
            "2PCB",
            "2PCC",
            "2SIC",
            "2VLJ",
            "2WPT",
            "3BK3",
            "3BN9",
            "3HFM",
            "3NPS",
            "3SGB",
            "4CPA",
        ]

    def get_raw_files(self):
        return [f"{self.root}/raw/{self.id}.pdb"]

    def download(self):
        meta_url = "https://life.bsc.es/pid/mutation_database/SKEMPI_1.1.csv"
        pdb_url = "https://life.bsc.es/pid/mutation_database/SKEMPI_pdbs.tar.gz"

        data_path = os.path.join(self.root, "raw", "SKEMPI_1.1.csv")
        pdb_path = os.path.join(self.root, "raw", "SKEMPI_pdbs.tar.gz")

        label_keys = ["Affinity_mut (M)", "Affinity_wt (M)"]
        mut_key = "Mutation(s)_cleaned"

        download_url(meta_url, data_path)
        download_url(pdb_url, pdb_path)
        extract_tar(
            os.path.join(self.root, "raw", "SKEMPI_pdbs.tar.gz"),
            os.path.join(self.root, "raw"),
        )

        def mut_index(mut_num, mut_chain, protein_dict):
            """Compute the index in the parsed sequence corresponding to the
            mutated residue from SKEMPI"""
            res_nums = protein_dict["residue"]["residue_number"]
            chain_ids = protein_dict["residue"]["chain_id"]
            try:
                hits = [
                    i
                    for i, (res, chain) in enumerate(zip(res_nums, chain_ids))
                    if str(res) == mut_num and chain == mut_chain
                ]
                return hits[0] + 1
            except IndexError:
                return None

        mut_clean = (
            lambda x, protein: x[0] + str(mut_index(x[2:-1], x[1], protein)) + x[-1]
        )
        df = pd.read_csv(data_path, sep=";")
        rows = []
        for protein, mutations in df.groupby("Protein"):
            if len(mutations) < 5:
                continue
            pdbid = protein.split("_")[0]
            protein_dict = self.parse_pdb(
                os.path.join(self.root, "raw", pdbid + ".pdb")
            )
            if protein_dict is None:
                continue

            mutations_col, y_col = [], []
            for _, mutant in mutations.iterrows():
                mutations = [
                    mut_clean(m, protein_dict) for m in mutant[mut_key].split(",")
                ]
                effect = np.log(mutant[label_keys[0]] / mutant[label_keys[1]])
                rows.append(
                    {
                        "pdbid": pdbid,
                        "sequence_wt": protein_dict["protein"]["sequence"],
                        "mutation": ":".join(mutations),
                        "effect": effect,
                    }
                )

                mutations_col.append(" ".join(mutations))
                y_col.append(effect)

            pd.DataFrame(rows).to_csv(os.path.join(self.root, "measurements.csv"))
            df = pd.DataFrame({"mutations": mutations_col, "y": y_col})
            df.to_csv(f"{self.root}/raw/{pdbid}.csv", index=False)


class SkempiDatasetV2(MutationDataset):
    @classmethod
    def available_ids(self):
        return [
            "1A22",
            "1A4Y",
            "1ACB",
            "1AHW",
            "1AK4",
            "1AO7",
            "1B41",
            "1BD2",
            "1BJ1",
            "1BP3",
            "1BRS",
            "1C1Y",
            "1C4Z",
            "1CBW",
            "1CHO",
            "1CSE",
            "1DAN",
            "1DQJ",
            "1DVF",
            "1E50",
            "1E96",
            "1EAW",
            "1EMV",
            "1F47",
            "1FC2",
            "1FCC",
            "1FFW",
            "1FSS",
            "1GC1",
            "1GCQ",
            "1GUA",
            "1H9D",
            "1HE8",
            "1IAR",
            "1JCK",
            "1JRH",
            "1JTD",
            "1JTG",
            "1K8R",
            "1KBH",
            "1KNE",
            "1KTZ",
            "1LFD",
            "1MAH",
            "1MHP",
            "1MI5",
            "1MLC",
            "1MQ8",
            "1N8Z",
            "1NMB",
            "1OGA",
            "1OHZ",
            "1PPF",
            "1QAB",
            "1R0R",
            "1REW",
            "1S1Q",
            "1SBB",
            "1TM1",
            "1UUZ",
            "1VFB",
            "1WQJ",
            "1XD3",
            "1YCS",
            "1YQV",
            "1YY9",
            "1Z7X",
            "2AJF",
            "2AK4",
            "2B0U",
            "2B2X",
            "2BDN",
            "2BNR",
            "2C5D",
            "2DVW",
            "2FTL",
            "2G2U",
            "2J0T",
            "2JEL",
            "2KSO",
            "2NY7",
            "2NYY",
            "2NZ9",
            "2O3B",
            "2PCB",
            "2PCC",
            "2SIC",
            "2VN5",
            "2WPT",
            "3AAA",
            "3B4V",
            "3BDY",
            "3BE1",
            "3BK3",
            "3BN9",
            "3BT1",
            "3BX1",
            "3C60",
            "3D3V",
            "3EQS",
            "3EQY",
            "3F1S",
            "3G6D",
            "3HFM",
            "3HG1",
            "3HH2",
            "3IDX",
            "3KBH",
            "3KUD",
            "3L5X",
            "3M62",
            "3M63",
            "3MZG",
            "3MZW",
            "3N06",
            "3N0P",
            "3N85",
            "3NCB",
            "3NCC",
            "3NGB",
            "3NPS",
            "3Q8D",
            "3QDG",
            "3QDJ",
            "3QHY",
            "3QIB",
            "3R9A",
            "3S9D",
            "3SE3",
            "3SE4",
            "3SE8",
            "3SE9",
            "3SEK",
            "3SF4",
            "3SGB",
            "3SZK",
            "3UIG",
            "3VR6",
            "4B0M",
            "4BFI",
            "4CPA",
            "4CVW",
            "4E6K",
            "4FTV",
            "4G0N",
            "4GNK",
            "4HFK",
            "4J2L",
            "4JEU",
            "4JFD",
            "4JFE",
            "4JFF",
            "4JPK",
            "4K71",
            "4L0P",
            "4L3E",
            "4MNQ",
            "4NM8",
            "4OFY",
            "4OZG",
            "4P23",
            "4P5T",
            "4PWX",
            "4RA0",
            "4RS1",
            "4UYP",
            "4UYQ",
            "4X4M",
            "4YFD",
            "4YH7",
            "5C6T",
            "5CXB",
            "5CYK",
            "5E6P",
            "5E9D",
            "5F4E",
            "5K39",
            "5M2O",
            "5TAR",
            "5UFE",
            "5UFQ",
            "5XCO",
        ]

    def get_raw_files(self):
        return [f"{self.root}/raw/PDBs/{self.id}.pdb"]

    def download(self):
        meta_url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
        pdb_url = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"

        data_path = os.path.join(self.root, "raw", "skempi_v2.csv")
        pdb_path = os.path.join(self.root, "raw", "SKEMPI2_PDBs.tgz")

        label_keys = ["Affinity_mut (M)", "Affinity_wt (M)"]
        mut_key = "Mutation(s)_cleaned"

        download_url(meta_url, data_path)
        download_url(pdb_url, pdb_path)
        extract_tar(pdb_path, os.path.join(self.root, "raw"))

        def mut_index(mut_num, mut_chain, protein_dict):
            """Compute the index in the parsed sequence corresponding to the
            mutated residue from SKEMPI"""
            res_nums = protein_dict["residue"]["residue_number"]
            chain_ids = protein_dict["residue"]["chain_id"]
            try:
                hits = [
                    i
                    for i, (res, chain) in enumerate(zip(res_nums, chain_ids))
                    if str(res) == mut_num and chain == mut_chain
                ]
                return hits[0] + 1
            except IndexError:
                return None

        mut_clean = (
            lambda x, protein: x[0] + str(mut_index(x[2:-1], x[1], protein)) + x[-1]
        )
        df = pd.read_csv(data_path, sep=";")
        df[label_keys[0]] = pd.to_numeric(df[label_keys[0]], errors="coerce")
        df[label_keys[1]] = pd.to_numeric(df[label_keys[1]], errors="coerce")
        df = df.dropna(subset=[label_keys[0], label_keys[1]])
        rows = []
        for protein, mutations in df.groupby("#Pdb"):
            if len(mutations) < 5:
                continue
            pdbid = protein.split("_")[0]
            protein_dict = self.parse_pdb(
                os.path.join(self.root, "raw", "PDBs", pdbid + ".pdb")
            )
            if protein_dict is None:
                continue

            protein_dict.to_csv(f"{self.root}/{pdbid}.csv")
            mutations_col, y_col = [], []
            for _, mutant in mutations.iterrows():
                mutations = [
                    mut_clean(m, protein_dict) for m in mutant[mut_key].split(",")
                ]
                effect = np.log(mutant[label_keys[0]] / mutant[label_keys[1]])
                rows.append(
                    {
                        "pdbid": pdbid,
                        "sequence_wt": protein_dict["protein"]["sequence"],
                        "mutation": ":".join(mutations),
                        "effect": effect,
                    }
                )

                mutations_col.append(" ".join(mutations))
                y_col.append(effect)

            pd.DataFrame(rows).to_csv(os.path.join(self.root, "measurements.csv"))
            df = pd.DataFrame({"mutations": mutations_col, "y": y_col})
            df.to_csv(f"{self.root}/raw/{pdbid}.csv", index=False)


class SNVDataset(MutationDataset):
    """Proteins from this paper:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0171355#pone.0171355.s004
    The 'y' attribute is 0 if mutation is neutral, 1 if it has an effect on the structure
    The ID for a dataset here contains two pdbids in the form <WT_PDBID>_<MUTANT_PDBID>
    """

    @classmethod
    def available_ids(self):
        return [
            "1L6X_1T83",
            "1L6X_1T89",
            "1J8U_1TDW",
            "4KY2_1TSH",
            "4KY2_1TTR",
            "4L7F_1UKH",
            "2WK6_1UOU",
            "1MSD_1VAR",
            "2NWD_1W08",
            "1PKW_1AGS",
            "1COH_1ABW",
            "4FF9_1AZV",
            "1LFG_1B0L",
            "1F41_1BZE",
            "2H7J_1NPZ",
            "1N46_1NQ0",
            "1N46_1NQ2",
            "1NIH_3NMM",
            "1GCJ_1O6O",
            "2QUG_1OPH",
            "2A2R_1PGT",
            "4A7U_1PTZ",
            "4OO6_1QBK",
            "3DLW_1QMN",
            "1LZJ_1R82",
            "1IK9_1FU1",
            "1VDG_1G0X",
            "1EK6_1HZJ",
            "1H0C_1J04",
            "3QR6_1K0M",
            "1LZ0_1LZ7",
            "1LZ0_1LZ7",
            "1LZ0_3IOH",
            "4GRN_1MIF",
            "4FF9_1N19",
            "1LCF_1N76",
            "2AOT_1JQE",
            "1AOS_1K62",
            "1LZ1_1LOZ",
            "1LZ0_1LZ7",
            "1EK6_1I3M",
            "1AUK_1E33",
            "4KY2_2G3X",
            "4H0W_2HAU",
            "1UCN_2HVE",
            "2IV5_3HEQ",
            "3POZ_2JIT",
            "2Q4G_2E0O",
            "3A8X_1ZRZ",
            "3A8X_1ZRZ",
            "1XT9_2BKR",
            "1X0V_1WPQ",
            "1XQG_1WZ9",
            "1XQG_1WZ9",
            "2DYP_1YDP",
            "1YJU_1YJR",
            "2FLU_1ZGK",
            "4N9O_3HAK",
            "3W7Y_1XW7",
            "4N9O_2LV1",
            "4DM9_4JKJ",
            "2F5J_2LKM",
            "2VNF_2M1R",
            "1LZJ_2O1F",
            "2OEX_2OJQ",
            "2ZT5_2PMF",
            "3B2T_2PZP",
            "2ZT5_2Q5I",
            "2CE2_2QUZ",
            "1P5F_2RK4",
            "4KY2_2TRH",
            "4KY2_2TRY",
            "4FF9_2VR6",
            "2WA5_2WA6",
            "2WA5_2WA7",
            "4FF9_2WYT",
            "1V5W_2ZJB",
            "4BMB_3AP5",
            "3PZD_3AU4",
            "2PVF_3B2T",
            "3MI9_3BLH",
            "3BWM_3BWY",
            "3PRX_3CU7",
            "3D6M_3D6H",
            "4KY2_3DJZ",
            "3NPC_3E7O",
            "2CVD_3EE2",
            "2X6G_3FPU",
            "1URO_3GW0",
            "1URO_3GW3",
            "4N9O_3HER",
            "4KY2_3I9A",
            "4DM9_3IRT",
            "3K7G_3K7J",
            "4FF9_3K91",
            "2YD0_3MDJ",
            "2YD0_3MDJ",
            "2YD0_3MDJ",
            "3LXP_3NYX",
            "2E2X_3P7Z",
            "4IGK_3PXA",
            "4IJ3_3Q6O",
            "3QE2_3QFC",
            "4FF9_3QQD",
            "1LS6_3QVU",
            "4A0P_3S2K",
            "3S4M_3S5E",
            "3TG4_3S7B",
            "4JBS_3SE6",
            "1LY7_3T3J",
            "1LY7_3T3K",
            "4LQD_3UB3",
            "4LQD_3UB4",
            "3GXB_3ZQK",
            "4FF9_4A7G",
            "1HBY_4AHD",
            "1HBY_4AHE",
            "1HBY_4AHG",
            "1HBY_4AHI",
            "1HBY_4AHJ",
            "1HBY_4AHK",
            "1HBY_4AHM",
            "2WOS_4AQJ",
            "4KSY_4F5D",
            "4FVP_4FVR",
            "3FXI_4G8A",
            "3FXI_4G8A",
            "4FU3_4HFG",
            "1F0Y_1F12",
            "1F13_1FIE",
            "1VDG_1G0X",
            "1WM5_1HH8",
            "1CZA_1HKC",
            "1OM9_1NA8",
            "1W6J_1W6K",
        ]

    def get_raw_files(self):
        return [
            os.path.join(self.root, "raw", f"{self.id}_{c}.pdb") for c in ["wt", "mut"]
        ]

    def download(self):
        STRUCTURES_URL = "https://doi.org/10.1371/journal.pone.0171355.s004"
        download_url(STRUCTURES_URL, os.path.join(self.root, "pdbs.csv"))

        df = pd.read_csv(os.path.join(self.root, "pdbs.csv"))
        df = df.dropna(subset=["PDBID", "WILDTYPE PDB ID"])
        # take the first if multiple PDBs given
        df["PDBID"] = df["PDBID"].apply(lambda x: x.split("/")[0])
        df["WILDTYPE PDB ID"] = df["WILDTYPE PDB ID"].apply(lambda x: x.split("/")[0])

        measurement_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                wt_id = row["WILDTYPE PDB ID"].strip()
                mut_id = row["PDBID"].strip()
                mut_pos = row["Residue in structure"]
                ppdb_wt = PandasPdb().fetch_pdb(wt_id)
                ppdb_mut = PandasPdb().fetch_pdb(mut_id)
            except Exception as e:
                print(e)
                continue

            entry_id = f"{wt_id}_{mut_id}"
            mut_pos_raw = int(row["Residue in structure"])
            mut_df = ppdb_mut.df["ATOM"]

            sequence_wt = ppdb_wt.amino3to1()
            sequence_mut = ppdb_mut.amino3to1()

            chains_wt = sequence_wt["chain_id"].unique()
            chains_mut = sequence_mut["chain_id"].unique()

            # only take single chain structures
            # if not ((len(chains_wt) == 1) and (len(chains_mut) == 1)):
            # continue

            sequence_wt = "".join(sequence_wt["residue_name"])
            sequence_mut = "".join(sequence_mut["residue_name"])

            # if len(sequence_wt) != len(sequence_mut):
            # continue

            wt_path = os.path.join(self.root, "raw", f"{entry_id}_wt.pdb")
            mut_path = os.path.join(self.root, "raw", f"{entry_id}_mut.pdb")

            ppdb_wt.to_pdb(path=wt_path, records=None, gz=False, append_newline=True)

            ppdb_mut.to_pdb(path=mut_path, records=None, gz=False, append_newline=True)

            # align the two structures
            tm_1, tm_2, rmsd, sup_df = tmalign_wrapper(
                mut_path, wt_path, return_superposition=True
            )

            wt_pdb_sup = copy.deepcopy(sup_df)
            mut_pdb_sup = copy.deepcopy(sup_df)

            wt_pdb_sup.df["ATOM"] = sup_df.df["ATOM"].loc[
                sup_df.df["ATOM"]["chain_id"] == "A"
            ]
            mut_pdb_sup.df["ATOM"] = sup_df.df["ATOM"].loc[
                sup_df.df["ATOM"]["chain_id"] == "B"
            ]

            mut_pdb_sup_ca = (
                mut_pdb_sup.df["ATOM"]
                .loc[mut_pdb_sup.df["ATOM"]["atom_name"] == "CA"]
                .reset_index()
            )
            wt_pdb_sup_ca = (
                wt_pdb_sup.df["ATOM"]
                .loc[wt_pdb_sup.df["ATOM"]["atom_name"] == "CA"]
                .reset_index()
            )

            try:
                mut_row = mut_pdb_sup_ca.loc[
                    mut_pdb_sup_ca["residue_number"] == mut_pos_raw
                ]
                mut_idx = mut_row.index[0] + 1
                mut_aa = AA_THREE_TO_ONE[mut_row.iloc[0]["residue_name"]]
                wt_aa = AA_THREE_TO_ONE[wt_pdb_sup_ca.iloc[mut_idx]["residue_name"]]
            except IndexError:
                print("skipping")
                continue

            wt_pdb_sup.to_pdb(path=wt_path, records=None, gz=False, append_newline=True)

            mut_pdb_sup.to_pdb(
                path=mut_path, records=None, gz=False, append_newline=True
            )

            effect = int(not bool(row["No structural consequence found"]))
            df = pd.DataFrame(
                {
                    "mutations": [wt_aa + str(mut_idx) + mut_aa],
                    "mutations_raw": [wt_aa + str(mut_pos_raw) + mut_aa],
                    "rmsd": rmsd,
                    "y": [effect],
                }
            )
            df.to_csv(os.path.join(self.root, "raw", f"{entry_id}.csv"))
            measurement_rows.append(
                {
                    "pdbid": entry_id,
                    "sequence_wt": "".join(sequence_wt),
                    "mutation": f"{wt_aa}{mut_idx}{mut_aa}",
                    "mutation_raw": f"{wt_aa}{mut_pos_raw}{mut_aa}",
                    "rmsd": rmsd,
                    "effect": effect,
                }
            )

        pd.DataFrame(measurement_rows).to_csv(
            os.path.join(self.root, "measurements.csv")
        )


if __name__ == "__main__":
    """
    for id in DeepSequenceDataset.available_ids():
        ds = DeepSequenceDataset(root="data/deepseq", id=id)
        mutations = ds.mutations
        ds = ds.to_graph(eps=8).pyg()
        graph, protein_dict = ds[0]
        print(len(ds))
        print(graph)
        print(mutations)
        break

    print(SkempiDatasetV2.available_ids())
    for mid in SkempiDatasetV2.available_ids():
        ds = SkempiDatasetV2(root="data/skempi", id=mid)
        mutations = ds.mutations
        ds = ds.to_graph(eps=8).pyg()
        graph, protein_dict = ds[0]
        print(mid, len(ds))
        print(graph)
        print(mutations)
    """
    for pid in SNVDataset.available_ids():
        strucs = SNVDataset(id=pid, use_precomputed=False).to_graph(eps=9).pyg()
