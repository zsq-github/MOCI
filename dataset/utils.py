import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from .config import cfg

def split_data(
        root: str,
        data: pd.DataFrame,
        train_size: float = cfg.train_size,
        seed: int = cfg.random_seed
) -> None:

    # get split index
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    # Split the data set into a training set and a test set
    train_idxs, test_idxs = list(split.split(data.filepath, data.label))[0]
    # train_idxs and test_idxs store the sample indexes of the split training and test sets, respectively.

    # get split result
    np_df = data.to_numpy()
    train_data = pd.DataFrame(np_df[train_idxs]).sort_values(by=0)
    # Ensure data is organized as it is stored
    test_data = pd.DataFrame(np_df[test_idxs]).sort_values(by=0)
    categories = np.unique(data.label.values)
    categories.sort()
    label_dict = {label: i for i, label in enumerate(categories)}

    # save result
    train_data.to_csv(os.path.join(root, 'CTScancustom_train.csv'), index=False, header=['filepath', 'label'])
    # Save training set data to a CSV file
    test_data.to_csv(os.path.join(root, 'CTScancustom_test.csv'), index=False, header=['filepath', 'label'])
    np.save(os.path.join(root, 'CTScancustom_dict.npy'), label_dict)
    # label_dict = np.load(os.path.join(root, 'custom_dict.npy'), allow_pickle='TRUE').item()
