from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torchvision import datasets, transforms
import copy
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import SVHNPolicy
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, \
    ToPILImage
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import pickle


class MY_SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, rate_partial=0.3, flag_aug=True
    ) -> None:
        super(MY_SVHN, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        self.flag_aug = flag_aug

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.targets = self.labels

        self.rate_partial = rate_partial

        self.partial_labels = self.generate_partial_labels()

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transform1 = Compose([
            ToTensor(),
            Cutout(n_holes=1, length=20),
            ToPILImage(),
            SVHNPolicy(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, partial_label = self.data[index], self.targets[index], self.partial_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.flag_aug:
            return img_ori, img1, img2, target, partial_label, index
        else:
            return img_ori, target, partial_label, index


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def generate_partial_labels(self):
        if (self.rate_partial != -1):
            def binarize_class(y):
                label = y.reshape(len(y), -1)
                enc = OneHotEncoder(categories='auto')
                enc.fit(label)
                label = enc.transform(label).toarray().astype(np.float32)
                label = torch.from_numpy(label)
                return label

                new_y = binarize_class(train_labels.clone())
                n, c = new_y.shape[0], new_y.shape[1]
                avgC = 0

            new_y = binarize_class(self.targets)
            n = len(self.targets)
            c = max(self.targets) + 1
            avgC = 0
            partial_rate = self.rate_partial
            print(partial_rate)
            for i in range(n):
                row = new_y[i, :]
                row[np.where(np.random.binomial(1, partial_rate, c) == 1)] = 1
                while torch.sum(row) == 1:
                    row[np.random.randint(0, c)] = 1
                avgC += torch.sum(row)
                new_y[i] = row

            avgC = avgC / n
            print("Finish Generating Candidate Label Sets:{}!\n".format(avgC))
            new_y = new_y.cpu().numpy()
            return new_y

        else:
            def binarize_class(y):
                label = y.reshape(len(y), -1)
                enc = OneHotEncoder(categories='auto')
                enc.fit(label)
                label = enc.transform(label).toarray().astype(np.float32)
                label = torch.from_numpy(label)
                return label

            def create_model(ds, feature, c):
                from partial_models.resnet import resnet
                from partial_models.mlp import mlp_phi
                if ds in ['kmnist', 'fmnist']:
                    net = mlp_phi(feature, c)
                elif ds in ['cifar10']:
                    net = resnet(depth=32, n_outputs=c)
                else:
                    pass
                return net

            def read_pkl(filename):
                res = []
                with open(filename, 'rb') as f:
                    while True:
                        try:
                            r = pickle.load(f)
                            res.append(r)
                        except EOFError:
                            break
                return res

            with torch.no_grad():
                c = max(self.targets) + 1
                data = torch.from_numpy(np.transpose(self.data, (0, 2, 3, 1)))
                y = binarize_class(torch.tensor(self.targets, dtype=torch.long))
                f = np.prod(list(data.shape)[1:])
                batch_size = 2000
                rate = 0.4
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                train_X, train_Y = data.to(device), y.to(device)
                train_X = train_X.permute(0, 3, 1, 2).to(torch.float32)
                train_p_Y_list = []
                step = train_X.size(0) // batch_size
                probs, _, _ = read_pkl('probs_svhn')
                if train_X.size(0) % batch_size != 0:
                    step += 1
                for i in range(0, step):
                    train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
                    partial_rate_array = torch.Tensor(probs[i * batch_size:(i + 1) * batch_size]).to(device)
                    partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
                    partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
                    partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
                    partial_rate_array[partial_rate_array > 1.0] = 1.0
                    m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
                    z = m.sample()
                    train_p_Y[torch.where(z == 1)] = 1.0
                    train_p_Y_list.append(train_p_Y)
                train_p_Y = torch.cat(train_p_Y_list, dim=0)
                print(train_p_Y.shape, train_X.shape)
                assert train_p_Y.shape[0] == train_X.shape[0]
            final_y = train_p_Y.cpu().clone()
            pn = final_y.sum() / torch.ones_like(final_y).sum()
            print("Partial type: instance dependent, Average Label: " + str(pn * c))
            return final_y.cpu().numpy()