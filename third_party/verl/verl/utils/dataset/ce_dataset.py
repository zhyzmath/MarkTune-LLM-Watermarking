"""Human-text data for the CE term in the joint GRPO update."""

from torch.utils.data import DataLoader, Dataset


class PlainTextDataset(Dataset):
    def __init__(self, files):
        import datasets

        frames = [datasets.load_dataset("parquet", data_files=path)["train"] for path in files]
        self.data = datasets.concatenate_datasets(frames)
        if "text" not in self.data.column_names:
            raise ValueError("CE Parquet input must contain a text column")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["text"]


def create_ce_train_loader(dataset, batch_size):
    if batch_size < 1:
        raise ValueError("CE batch size must be positive")
    if len(dataset) < batch_size:
        raise ValueError(f"CE training requires at least {batch_size} documents, found {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
