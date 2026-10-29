"""Frozen paired-brightfield data and an offset-addressable sample stream."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Revvity(Dataset):
    def __init__(self, manifest, transform=None):
        self.manifest = Path(manifest)
        self.transform = transform
        with self.manifest.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'sample_id', 'bf_higher', 'bf_lower'}.issubset(reader.fieldnames or []):
                raise ValueError('Missing paired-brightfield manifest columns')
            self.records = list(reader)
        ids = [r['sample_id'] for r in self.records]
        if not ids or not all(ids) or len(set(ids)) != len(ids):
            raise ValueError('Expected nonempty unique field IDs')

    def __len__(self):
        return len(self.records)

    def validate_files(self):
        for row in self.records:
            for key in ('bf_higher', 'bf_lower'):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])

    def get_image(self, index):
        row = self.records[index]
        planes = []
        for key in ('bf_higher', 'bf_lower'):
            with Image.open(row[key]) as image:
                planes.append(np.array(image, dtype=np.uint16))
        high, low = planes
        if high.ndim != 2 or high.shape != low.shape:
            raise ValueError(f"Invalid paired planes: {row['sample_id']}")
        maximum = max(int(high.max()), int(low.max()))
        if maximum == 0:
            raise ValueError(f"Empty paired planes: {row['sample_id']}")
        rgb = np.stack([high, high, low], axis=-1).astype(np.float32)
        return Image.fromarray((rgb * (255.0 / maximum)).astype(np.uint8))

    def __getitem__(self, index):
        image = self.get_image(index)
        return self.transform(image) if self.transform is not None else image
