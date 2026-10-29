"""Dataset selection; method transforms and training do not own dataset schemas."""

from src.data.datasets.revvity import Revvity
from src.run.artifacts import sha256


DATASETS = {'REVVITY': Revvity}


def make_dataset(config, transform=None):
    try:
        dataset_type = DATASETS[config.name]
    except KeyError:
        raise ValueError(f'Unknown dataset {config.name!r}; available: {sorted(DATASETS)}') from None
    return dataset_type(config.manifest, transform)


def validate_data(config, check_files=True):
    dataset = make_dataset(config)
    if sha256(config.manifest) != config.manifest_sha256 or len(dataset) != config.expected_fields:
        raise ValueError('Frozen manifest changed')
    if check_files:
        dataset.validate_files()
    return dict(path=config.manifest, sha256=config.manifest_sha256,
                fields=len(dataset), version=config.version)
