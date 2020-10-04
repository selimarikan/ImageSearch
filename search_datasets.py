import pathlib

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class SearchDataset(VisionDataset):
    def __init__(self, root_path: str, transform):
        """
        A very simple folder-image dataset that checks for JPG files
        """
        super(SearchDataset, self).__init__(root_path, transform=transform)
        self.image_paths = [
            str(p) for p in list(pathlib.Path(root_path).rglob("*.jpg"))
        ]
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, path) where path is the image file path.
        """
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.image_paths)
