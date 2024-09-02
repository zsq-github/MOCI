from torch.utils.data import DataLoader
from torchvision import transforms
from .chestXray import CXR
from .ctScan import CTScan
from .chestXTwo import CXRTwo
from .config import cfg

# Get the data loader for the training and test sets
def get_loader(dataset, batch_size, num_workers=4, transform_train=None, transform_test=None):
    root = cfg.default_root
    if dataset == 'CT_scan':
        #Dataset =CXRTwo
        Dataset = CTScan

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    else:
        raise NotImplementedError

    data_train = Dataset(root=root,
                          transform=transform_train)
    data_test = Dataset(root=root,
                         train=False,
                         transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

    # Returns data loaders for training and test sets
    return data_train_loader, data_test_loader
