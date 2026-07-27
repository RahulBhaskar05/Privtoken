"""
Market-1501 Dataset, PK Sampler, and DataLoader utilities.

Provides:
- Market1501 Dataset class with proper PID parsing/remapping
- PKSampler for triplet-loss-friendly batching
- Train/test transforms following standard Re-ID augmentation
- get_dataloader() factory function
"""

import os
import re
import glob
import random
import copy
from collections import defaultdict
import json

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms


class Market1501(Dataset):
    """
    Market-1501 dataset loader.

    Args:
        root (str): Root directory of Market-1501 (contains bounding_box_train/, etc.)
        split (str): One of 'train', 'gallery', 'query'.
        transform (callable, optional): Image transform pipeline.

    Returns per __getitem__:
        image (Tensor): Transformed image, shape (3, H, W).
        pid (int): Person identity (0-indexed for train, original for gallery/query).
        camid (int): Camera ID (0-indexed).
        img_path (str): Full path to the image file.
    """

    _SPLIT_DIRS = {
        'train': 'bounding_box_train',
        'gallery': 'bounding_box_test',
        'query': 'query',
    }

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        assert split in self._SPLIT_DIRS, f"split must be one of {list(self._SPLIT_DIRS.keys())}"
        self.root = root
        self.split = split
        self.transform = transform

        data_dir = os.path.join(root, self._SPLIT_DIRS[split])
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}\n"
                f"Expected Market-1501 structure:\n"
                f"  {root}/\n"
                f"    ├── bounding_box_train/\n"
                f"    ├── bounding_box_test/\n"
                f"    ├── query/\n"
                f"    ├── gt_bbox/\n"
                f"    └── gt_query/"
            )

        # Collect all valid .jpg images
        img_paths = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
        if len(img_paths) == 0:
            raise RuntimeError(f"No .jpg images found in {data_dir}")

        # Parse filenames and build dataset
        self.data = []  # list of (img_path, pid, camid)
        pid_set = set()

        for path in img_paths:
            fname = os.path.basename(path)
            # Filename format: PPPP_cCsSS_FFFFFF_DD.jpg or -1_c1s1_...
            parts = fname.split('_')
            pid = int(parts[0])
            camid = int(parts[1][1]) - 1  # 0-indexed camera ID

            if pid == -1 and split == 'train':
                continue  # skip junk/distractor images in training
            self.data.append((path, pid, camid))
            pid_set.add(pid)

        # For training split: remap PIDs to contiguous 0-indexed integers
        if split == 'train':
            sorted_pids = sorted(pid_set)
            self._pid_to_label = {pid: label for label, pid in enumerate(sorted_pids)}
            self.data = [
                (path, self._pid_to_label[pid], camid)
                for path, pid, camid in self.data
            ]
            self.num_classes = len(sorted_pids)
        else:
            self.num_classes = len(pid_set)

        # Build pid-to-indices mapping (needed for PKSampler)
        self.pid_to_indices = defaultdict(list)
        for idx, (_, pid, _) in enumerate(self.data):
            self.pid_to_indices[pid].append(idx)
        self.pids = sorted(self.pid_to_indices.keys())

        # Store camids as accessible attribute for evaluation
        self.camids = [camid for _, _, camid in self.data]

        print(f"[Market1501] Loaded {split}: {len(self.data)} images, "
              f"{len(pid_set)} identities")

    def __getitem__(self, index):
        """
        Returns:
            image (Tensor): shape (3, H, W)
            pid (int): person identity
            camid (int): camera id (0-indexed)
            img_path (str): absolute path to image
        """
        img_path, pid, camid = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path

    def __len__(self):
        return len(self.data)


class PKSampler(Sampler):
    """
    Person-Key (PK) batch sampler for triplet loss training.

    Each batch contains exactly `num_instances` images from each of
    `batch_size // num_instances` randomly selected identities, guaranteeing
    positive pairs for hard triplet mining.

    Args:
        dataset (Market1501): Dataset with pid_to_indices and pids attributes.
        num_instances (int): Number of images per identity per batch.
        batch_size (int): Total batch size (must be divisible by num_instances).
    """

    def __init__(self, dataset, num_instances, batch_size):
        self.pid_to_indices = copy.deepcopy(dataset.pid_to_indices)
        self.pids = dataset.pids
        self.num_instances = num_instances
        self.batch_size = batch_size

        assert batch_size % num_instances == 0, \
            f"batch_size ({batch_size}) must be divisible by num_instances ({num_instances})"
        self.num_pids_per_batch = batch_size // num_instances

        # Estimate total number of samples per epoch
        self._length = 0
        for pid in self.pids:
            num_imgs = len(self.pid_to_indices[pid])
            self._length += max(num_imgs, self.num_instances)

    def __iter__(self):
        """
        Yields batch_size indices at a time, each batch containing
        num_instances images from num_pids_per_batch identities.
        """
        # Build index dictionary with shuffled copies
        index_dict = {}
        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_to_indices[pid])
            if len(idxs) < self.num_instances:
                # Oversample if not enough images for this identity
                idxs = idxs * (self.num_instances // len(idxs) + 1)
            random.shuffle(idxs)
            index_dict[pid] = idxs

        avail_pids = copy.deepcopy(self.pids)
        random.shuffle(avail_pids)

        batch = []
        pid_idx = 0

        expected_batches = self.__len__()
        yielded_batches = 0

        while pid_idx < len(avail_pids):
            pid = avail_pids[pid_idx]
            pid_idx += 1

            # Get num_instances images for this pid
            if len(index_dict[pid]) < self.num_instances:
                # Refill if exhausted
                idxs = copy.deepcopy(self.pid_to_indices[pid])
                if len(idxs) < self.num_instances:
                    idxs = idxs * (self.num_instances // len(idxs) + 1)
                random.shuffle(idxs)
                index_dict[pid] = idxs

            selected = index_dict[pid][:self.num_instances]
            index_dict[pid] = index_dict[pid][self.num_instances:]
            batch.extend(selected)

            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                yielded_batches += 1
                batch = batch[self.batch_size:]
                if yielded_batches >= expected_batches:
                    return

            # If we've gone through all pids, reshuffle and start over
            if pid_idx >= len(avail_pids):
                if len(batch) > 0:
                    # try to fill the rest of the batch by reshuffling pids
                    avail_pids = copy.deepcopy(self.pids)
                    random.shuffle(avail_pids)
                    pid_idx = 0
                    # If we can now make at least one full batch, yield as many as possible
                    while len(batch) >= self.batch_size:
                        yield batch[:self.batch_size]
                        yielded_batches += 1
                        batch = batch[self.batch_size:]
                        if yielded_batches >= expected_batches:
                            return
                    # If not enough to form another full batch and no more pids to add, stop
                    if len(avail_pids) < (self.batch_size - len(batch)) // self.num_instances:
                        break

    def __len__(self):
        return self._length // self.batch_size


class CUHK03ClassicDataset(Dataset):
    """
    CUHK03 Classic protocol dataset loader.

    Expects a split JSON like `splits_classic_labeled.json` with structure:
        [ { "train": [[path,pid,camid], ...], "query": [...], "gallery": [...] } ]

    Paths in the JSON may be like "./data\\cuhk03\\images_labeled\\1_001_1_01.png".
    We normalize and resolve them against `root` + `images_dir`.
    """

    def __init__(self, root, split='train', split_file=None, images_dir='images_labeled', transform=None):
        super().__init__()
        assert split in ('train', 'query', 'gallery')
        self.root = root
        self.split = split
        self.transform = transform
        self.images_dir = images_dir

        if split_file is None:
            raise ValueError('split_file must be provided for CUHK03ClassicDataset')
        # Resolve split_file: try dataset root, parent of root, then cwd
        candidates = []
        if os.path.isabs(split_file):
            candidates.append(split_file)
        else:
            candidates.append(os.path.normpath(os.path.join(root, split_file)))
            candidates.append(os.path.normpath(os.path.join(os.path.dirname(root), split_file)))
            candidates.append(os.path.normpath(os.path.join(os.getcwd(), split_file)))

        found = None
        for c in candidates:
            if os.path.isfile(c):
                found = c
                break
        if found is None:
            raise FileNotFoundError(f"CUHK03 split file not found. Tried: {candidates}")
        split_file = found

        with open(split_file, 'r', encoding='utf-8') as f:
            content = json.load(f)

        if isinstance(content, list) and len(content) > 0:
            splits = content[0]
        elif isinstance(content, dict):
            splits = content
        else:
            raise RuntimeError('Unexpected split file format')

        if split not in splits:
            raise KeyError(f"Split '{split}' not found in split file")

        entries = splits[split]

        # If the JSON provides query/gallery, collect test PIDs to exclude from training
        test_pids = set()
        if 'query' in splits and 'gallery' in splits:
            test_pids.update(pid for _, pid, _ in splits['query'])
            test_pids.update(pid for _, pid, _ in splits['gallery'])

        self.data = []
        pid_set = set()

        missing = []
        for entry in entries:
            if not (isinstance(entry, list) or isinstance(entry, tuple)) or len(entry) < 3:
                continue
            pstr, pid, camid = entry[0], int(entry[1]), int(entry[2])
            # Skip entries belonging to test PIDs when building the training split
            if split == 'train' and pid in test_pids:
                continue

            # Normalize and extract basename
            pnorm = os.path.normpath(pstr.replace('./', '').lstrip('.\\/'))
            basename = os.path.basename(pnorm)

            # Try multiple candidate roots for image location
            img_path = None
            candidate_roots = [root, os.path.dirname(root), os.getcwd()]
            for base in candidate_roots:
                candidate = os.path.normpath(os.path.join(base, images_dir, basename))
                if os.path.isfile(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                missing.append(os.path.normpath(os.path.join(root, images_dir, basename)))
                continue

            # Convert camid to 0-index for consistency
            camid0 = camid - 1

            self.data.append((img_path, pid, camid0))
            pid_set.add(pid)

        if len(missing) > 0:
            # Report first missing path to help debugging
            raise FileNotFoundError(f"Missing {len(missing)} images; example missing: {missing[0]}")

        # For training split: remap PIDs to contiguous 0..N-1
        if split == 'train':
            sorted_pids = sorted(pid_set)
            self._pid_to_label = {pid: label for label, pid in enumerate(sorted_pids)}
            self.data = [
                (path, self._pid_to_label[pid], camid)
                for path, pid, camid in self.data
            ]
            self.num_classes = len(sorted_pids)
        else:
            self.num_classes = len(pid_set)

        # Build pid-to-indices mapping
        self.pid_to_indices = defaultdict(list)
        for idx, (_, pid, _) in enumerate(self.data):
            self.pid_to_indices[pid].append(idx)
        self.pids = sorted(self.pid_to_indices.keys())
        self.camids = [camid for _, _, camid in self.data]

        print(f"[CUHK03Classic] Loaded {split}: {len(self.data)} images, {len(pid_set)} identities")

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path

    def __len__(self):
        return len(self.data)


def get_train_transforms(img_h, img_w):
    """
    Training augmentation pipeline following standard Re-ID practice.

    Args:
        img_h (int): Target image height (e.g. 256).
        img_w (int): Target image width (e.g. 128).

    Returns:
        transforms.Compose: Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((img_h, img_w)),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    ])


def get_test_transforms(img_h, img_w):
    """
    Test/evaluation transform pipeline (no augmentation).

    Args:
        img_h (int): Target image height (e.g. 256).
        img_w (int): Target image width (e.g. 128).

    Returns:
        transforms.Compose: Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloader(cfg, split):
    """
    Create a DataLoader for the specified split.

    Args:
        cfg (dict): Configuration dictionary with keys: data_root, img_height,
                     img_width, batch_size, num_instances, num_workers.
        split (str): One of 'train', 'gallery', 'query'.

    Returns:
        DataLoader: PyTorch DataLoader for the split.
        Market1501: The underlying dataset instance.
    """
    if split == 'train':
        transform = get_train_transforms(cfg['img_height'], cfg['img_width'])
    else:
        transform = get_test_transforms(cfg['img_height'], cfg['img_width'])

    # Support CUHK03 Classic protocol via config key `dataset_name: cuhk03_classic`.
    dataset_name = cfg.get('dataset_name', cfg.get('dataset', 'market1501')).lower()
    if dataset_name in ('cuhk03_classic', 'cuhk03classic', 'cuhk03'):
        # Read expected keys: dataset_root, split_file, images_dir
        dataset_root = cfg.get('dataset_root', cfg.get('data_root'))
        if dataset_root is None:
            raise KeyError('dataset_root or data_root must be set in config for CUHK03')
        split_file = cfg.get('split_file', os.path.join(dataset_root, 'splits_classic_labeled.json'))
        images_dir = cfg.get('images_dir', 'images_labeled')

        dataset = CUHK03ClassicDataset(
            root=dataset_root,
            split=split,
            split_file=split_file,
            images_dir=images_dir,
            transform=transform,
        )
        # Ensure num_classes is available to model init
        if split == 'train':
            cfg['num_classes'] = dataset.num_classes
    else:
        dataset = Market1501(
            root=cfg['data_root'],
            split=split,
            transform=transform,
        )

    if split == 'train':
        batch_sampler = PKSampler(dataset, cfg['num_instances'], cfg['batch_size'])
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg['num_workers'],
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

    return loader, dataset
