from joblib import dump
import json
import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import torch.cuda
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

import search_datasets


def read_config() -> dict:
    """
    Read config from current directory config.json
    """
    filename: str = "config.json"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist.")

    with open(filename) as f:
        config = json.load(f)

    return config


def get_device():
    has_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {has_cuda}")
    device = torch.device("cuda:0" if has_cuda else "cpu")


def get_model():
    """
    A Pretrained ResNet34 is used for now
    """
    net = models.resnet34(pretrained=True, progress=True)
    # Replace last layer so that we get the features rather than class probs
    net.fc = nn.Identity(net.fc.in_features)
    net.eval()

    return net


def get_trafo(input_size: tuple, dataset_mean: list, dataset_std: list):
    trafo = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ]
    )

    return trafo


def extract_single_image(path: str, trafo: transforms.Compose, model) -> np.array:
    pil_image = default_loader(path)
    return extract_single_image_PIL(pil_image, trafo, model)


def extract_single_image_PIL(
    pil_image: Image, trafo: transforms.Compose, model
) -> np.array:
    tensor = trafo(pil_image)
    tensor.unsqueeze_(0)
    return model(tensor).cpu().detach().numpy()


def main():
    config = read_config()
    # ImageNet vals
    dataset_mean = config["model"]["dataset_mean"]
    dataset_std = config["model"]["dataset_std"]

    input_size = (config["model"]["input_size"], config["model"]["input_size"])
    batch_size = config["model"]["batch_size"]
    images_dir = config["extractor"]["input_dir"]

    net = get_model()
    trafo = get_trafo(input_size, dataset_mean, dataset_std)
    device = get_device()

    dataset = search_datasets.SearchDataset(images_dir, transform=trafo)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    print(f"Input Image Dir: {images_dir}")
    print(f"Image Count: {len(dataset)}")

    # Move model to GPU
    net = net.to(device)

    vectors = []
    image_paths = []

    epoch_count = int(len(dataset) / batch_size)
    counter = 0

    for inputs, paths in dataloader:
        print(f"{counter}/{epoch_count} New iteration")
        counter = counter + 1

        # Move images to GPU
        inputs = inputs.to(device)

        # Do a forward pass
        outputs = net(inputs)

        vectors.extend(outputs.cpu().detach().numpy())
        image_paths.extend(paths)

    df = pd.DataFrame(vectors)
    df.index = image_paths
    # Parquet needs string columns
    df.columns = df.columns.astype(str)
    df.to_parquet(config["extractor"]["out_features"])

    # Construct the nearest model
    nbrs = NearestNeighbors(
        n_neighbors=config["nearest"]["k_nearest"], algorithm="ball_tree"
    ).fit(df)
    # Save to disk
    dump(nbrs, config["nearest"]["model_path"])

    print("DONE")


if __name__ == "__main__":
    main()