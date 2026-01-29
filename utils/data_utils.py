from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader


class TargetReferenceDataset(Dataset):
    def __init__(self, root, weights: dict[str, float] | None = None):
        """
        weights: optional mapping {filename.json: weight}, e.g. {"special.json": 5.0}
                 filenames not in the dict get weight 1.0
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

        # Build an index list with repetitions according to weights
        weights = weights or {}
        self._index = []
        for i, (fname, _, _) in enumerate(self.samples):
            w = float(weights.get(fname, 1.0))
            if w <= 0:
                continue  # skip if weight is 0
            reps = max(1, int(round(w)))  # duplicate count
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