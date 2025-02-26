from torch.utils.data import Dataset


class DictDatasetWrapper(Dataset):
    """Wraps a dataset that returns tuples into one that returns dictionaries."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        numeric, categorical = self.dataset[idx]
        return {"numeric": numeric, "categorical": categorical}
