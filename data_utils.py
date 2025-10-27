from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader


class TargetReferenceDataset(Dataset):
    def __init__(self, root, *, neutral_name: str = "neutral.json", neutral_mult: float = 8.0):
        """
        Oversample a single file (neutral_name) by neutral_mult.
        All other files have weight 1.0 (uniform among themselves).
        """
        self.files = sorted(Path(root).glob("*.json"))
        self.samples = []
        for fp in self.files:
            try:
                text = fp.read_text(encoding="utf-8-sig")
                obj = json.loads(text)
                t = (obj.get("target") or "").strip()
                r = (obj.get("reference") or "").strip()
                self.samples.append((fp.name, t, r))
            except json.JSONDecodeError as e:
                print(f"BAD JSON {fp}: {e}")
                print("Head:", repr(text[:120]))
                continue
        if not self.samples:
            raise RuntimeError("No valid items found.")

        # Build an index list with repetitions:
        #   neutral_name -> weight = neutral_mult
        #   others       -> weight = 1.0
        self._index = []
        for i, (fname, _, _) in enumerate(self.samples):
            w = neutral_mult if fname == neutral_name else 1.0
            reps = max(1, int(round(w)))
            self._index.extend([i] * reps)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        i = self._index[idx]
        fname, target, reference = self.samples[i]
        return {"file": fname, "target": target, "reference": reference}


# simple collate that keeps strings
def collate_prompts(batch):
    return {
        "file": [b["file"] for b in batch],
        "target": [b["target"] for b in batch],
        "reference": [b["reference"] for b in batch],
    }








if __name__ == "__main__":
    data_dir = "/Users/piotrwojcik/PycharmProjects/UnGuide/data_small"  # <-- change me

    ds = TargetReferenceDataset(data_dir)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_prompts)

    for batch in loader:
        print(batch['file'])
        print(batch['target'])
        print(batch['reference'])
        print('----')