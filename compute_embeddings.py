import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import check_call

import numpy as np
import torch
from torch.nn import Conv2d, Identity, Module
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--encoder", type=str, default="simclr")

    parser.add_argument("--batch_size", type=int, default=1_024)

    return parser.parse_args()


def get_device(use_gpu: bool = True, use_deterministic_ops: bool = False) -> str:
    """
    References:
        https://pytorch.org/docs/stable/notes/mps.html
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if use_deterministic_ops:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return device


def get_encoder_state_dict(dataset: str, encoder: str, model_dir: Path) -> dict:
    file_ids = {
        # https://github.com/vturrisi/solo-learn/blob/main/zoo/cifar10.sh
        "cifar10": {
            "barlow": "1x7y44E05vuobibfObT4n3jqLI8QNVESV",
            "byol": "1zOE8O2yPyhE23LMoesMoDPdLyh1qbI8k",
            "deepcluster": "13L_QlwrBRJhdeCaVdgkRYWfvoh4PIWwj",
            "dino": "1Wv9w5j22YitGAWi4p3IJYzLVo4fQkpSu",
            "mocov2": "1viIUTHmLdozDWtzMicV4oOyC50iL2QDU",
            "mocov3": "1EFHWBLYFsglZYPYsBc0YrtihrzBZRe7h",
            "nnclr": "1zKReUmJ35vRnQxfSxn7yRVRW_oy3LUDF",
            "ressl": "1UdDWvgpyvj3VFVm0lq-WrGj0-GTcEpHq",
            "simclr": "15fI7gb9M92jZWBZoGLvarYDiNYK3RN2O",
            "simsiam": "1ZMGGTziK0DbCP43fDx2rPFrtJxCLJDmb",
            "supcon": "1tkk_r7tYozLgf9khW6LiGxaTvJQ4c5sA",
            "swav": "1CPok55wwN_4QecEjubdLeBo_9qWSJTHw",
            "vibcreg": "1dHsKrhCcwWIXFwQJ4oVPgLcEcT3SecQV",
            "vicreg": "1TeliMNt5bOchqJj2u_JjB0_ahKB5LKi5",
            "wmse": "1jTjpmVTi9rtzy3NPEEp_61py-jeHy5fi",
        },
        # https://github.com/vturrisi/solo-learn/blob/main/zoo/cifar100.sh
        "cifar100": {
            "barlow": "17cZt3DorfiCYb0ZauLHv0iM-YDGYa-mE",
            "byol": "1fE7TdRboFJnYXr8JSY_tGmuFGitI8l23",
            "deepcluster": "1grFfh0aaVYpeuYbgFYB4rmfj9uvXhYSd",
            "dino": "16gdp5L_a9BVcRvcU4f-NUJCsIpX3Oecr",
            "mocov2": "1KNkCA2Hr70QsmOSif9_UUndFerOb7Jft",
            "mocov3": "1QAuKJmegGCJrntAL80tfTrbi2fI4sPl-",
            "nnclr": "1aodwBlGK6EqrC_kthk8JcuxVcY4S5CF9",
            "ressl": "16sKNdpScv5FckpC02W41mjETXL6T5u2S",
            "simclr": "17YGC7y4yxkVAF8ZNezdtmN-uc70jz3zq",
            "simsiam": "1DStn9PAEMJtzh1Mxb3NjfTtm5vaNgRM5",
            "supcon": "1QhPHENtgYttIF1Dn1srA4dAkIiC_5P7W",
            "swav": "1oJzFfayNpcShK1bZtDK58HthcKY2bpns",
            "vibcreg": "1akNcewHzh4ideoQPWakaXWGDxfGoxkNu",
            "vicreg": "1kH78BUBKprrsxL2KRKmorVQ9vJHsMsID",
            "wmse": "1_6EmYFqAW_U8DFv72KUaAe-BV8xkRxsp",
        },
    }

    state_dict_paths = list(model_dir.glob(f"{encoder}*.ckpt"))

    if len(state_dict_paths) == 0:
        file_id = file_ids[dataset][encoder]

        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir_str = f"{model_dir}{os.sep}"

        check_call(["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", model_dir_str])

    state_dict_path = list(model_dir.glob(f"{encoder}*.ckpt"))[0]

    return torch.load(state_dict_path)["state_dict"]


@torch.inference_mode()
def encode(loader: DataLoader, encoder: Module, device: str) -> np.ndarray:
    embeddings = []

    for images_i, _ in loader:
        images_i = images_i.to(device)
        embeddings_i = encoder(images_i)
        embeddings += [embeddings_i.cpu().numpy()]

    return np.concatenate(embeddings)


def main(cfg: Namespace) -> None:
    """
    References:
        https://github.com/vturrisi/solo-learn/blob/main/docs/source/tutorials/offline_linear_eval.rst
    """
    device = get_device(cfg.use_gpu)

    data_dir = Path(cfg.data_dir) / cfg.dataset
    model_dir = Path(cfg.model_dir) / cfg.dataset

    state_dict = get_encoder_state_dict(cfg.dataset, cfg.encoder, model_dir)

    # Use list(state_dict.keys()) instead of state_dict.keys() so that we can delete keys on the go.
    for key in list(state_dict.keys()):
        if "backbone" in key:
            state_dict[key.replace("backbone.", "")] = state_dict[key]

        del state_dict[key]

    encoder = resnet18()
    encoder.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    encoder.maxpool = Identity()
    encoder.fc = Identity()
    encoder = encoder.to(device)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    for subset in ("train", "test"):
        dataset_class = CIFAR10 if cfg.dataset == "cifar10" else CIFAR100

        dataset = dataset_class(
            root=data_dir,
            train=(subset == "train"),
            download=True,
            transform=Compose(
                [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])]
            ),
        )

        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

        embeddings = encode(loader, encoder, device)

        np.save(data_dir / f"embeddings_{cfg.encoder}_{subset}.npy", embeddings, allow_pickle=False)
        np.save(data_dir / f"labels_{subset}.npy", dataset.targets, allow_pickle=False)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
