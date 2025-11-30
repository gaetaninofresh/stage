import pyarrow as pa
import pyarrow.feather as feather
import torch


class DecompDB:

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

        n = len(labels)
        for k, v in encodings.items():
            if len(v) != n:
                raise ValueError(f"Encoding field '{k}' has length {
                                 len(v)} but expected {n}")

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def append(self, encodings, labels):

        is_single = isinstance(labels, (int, float)
                               ) or not hasattr(labels, "__len__")

        if is_single:
            labels = [labels]
            encodings = {k: [v] for k, v in encodings.items()}

        batch_size = len(labels)

        for k in self.encodings.keys():
            if k not in encodings:
                raise KeyError(
                    f"append(): missing encoding field '{k}' in input")
            if len(encodings[k]) != batch_size:
                raise ValueError(
                    f"append(): field '{k}' length {
                        len(encodings[k])}, expected {batch_size}"
                )

        for k in self.encodings.keys():
            self.encodings[k].extend(encodings[k])

        self.labels.extend(labels)

    def to_arrow_table(self):
        data = {}
        schema = pa.schema({
            'ids': pa.list_(pa.uint64()),
            'attention_mask': pa.list_(pa.uint64()),
            "labels": pa.uint64()
        })
        for key, values in self.encodings.items():
            data[key] = pa.array(values)
            data["labels"] = self.labels

        return pa.table(data, schema)

    def save_arrow(self, path: str):
        table = self.to_arrow_table()
        feather.write_feather(table, path)
