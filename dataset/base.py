import os
import warnings
from typing import Optional, Callable, Any, Tuple
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset

# Load the image dataset for the classification task
class ClassificationVisionDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,  # Converting Images
                 target_transform: Optional[Callable] = None,  # Preprocess the loaded target data
                 loader: Optional[Callable] = None,  # Loading image data
    ) -> None:

        super(ClassificationVisionDataset, self).__init__(root, transform=transform,
                                                          target_transform=target_transform)

        if not self._check_split():
            warnings.warn('split files not found or corrupted, re-splitting the dataset')
            self.split_data()  # Implementing Segmentation Operations on Data Sets

        # Load Category Dictionary File
        self.class_to_idx: dict = np.load(os.path.join(root, 'CTScancustom_dict.npy'), allow_pickle=True).item()
        self.categories = list(self.class_to_idx.keys())

        if train:
            samples = pd.read_csv(os.path.join(root, 'CTScancustom_train.csv'))
        else:
            samples = pd.read_csv(os.path.join(root, 'CTScancustom_test.csv'))

        samples.filepath = root + '/' + samples.filepath
        self.samples = samples.to_numpy().tolist()
        self.loader = loader if loader is not None else default_loader
        # Setting the Loader Function
    def split_data(self) -> None:
        raise NotImplementedError

    # Get the images and captions at the specified index position in the record
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]
        target = self.class_to_idx[target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # Get the total number of samples in the data set
    def __len__(self) -> int:
        return len(self.samples)

    def _check_split(self) -> bool:
        # check the existence of split files
        check_list = ['CTScancustom_train.csv', 'CTScancustom_test.csv', 'CTScancustom_dict.npy']
        for file in check_list:
            if not os.path.exists(os.path.join(self.root, file)):
                return False
        return True
