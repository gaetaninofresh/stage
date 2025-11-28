
import torch


class DecompDB:
    """
    Flexible dataset with unified append() for both single and batch additions.
    """

    def __init__(self, ids, code_clean, code_processed, encodings, labels):
        self.ids = ids
        self.code_clean = code_clean
        self.code_processed = code_processed

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

    # -------------------------------------------------------
    # Unified single-or-batch append
    # -------------------------------------------------------
    def append(self, ids, code_clean, code_processed, encodings, labels):
        """
        Append either a single sample OR a batch of samples.

        If inputs are scalars (str, dict, int), treats as a single sample.
        If inputs are lists, treats as batch append.

        Args:
            ids: str or list[str]
            code_clean: str or list[str]
            code_processed: str or list[str]
            encodings: dict or dict[str, list]
            labels: int/float or list
        """

        # ------------------------------
        # Detect single vs batch
        # ------------------------------
        is_single = isinstance(labels, (int, float)
                               ) or not hasattr(labels, "__len__")

        if is_single:
            # Wrap everything into lists
            ids = [ids]
            code_clean = [code_clean]
            code_processed = [code_processed]
            labels = [labels]
            encodings = {k: [v] for k, v in encodings.items()}

        # Now guaranteed batch mode:
        batch_size = len(labels)

        # ------------------------------
        # Validate input lengths
        # ------------------------------
        if not (len(ids) == len(code_clean) == len(code_processed) == batch_size):
            raise ValueError("append(): mismatched list lengths in batch.")

        for k in self.encodings.keys():
            if k not in encodings:
                raise KeyError(f"append(): missing encoding field '{k}'")
            if len(encodings[k]) != batch_size:
                raise ValueError(
                    f"append(): field '{k}' has {
                        len(encodings[k])}, expected {batch_size}"
                )

        # ------------------------------
        # Append data
        # ------------------------------
        self.ids.extend(ids)
        self.code_clean.extend(code_clean)
        self.code_processed.extend(code_processed)
        self.labels.extend(labels)

        for k in self.encodings.keys():
            self.encodings[k].extend(encodings[k])
