from pathlib import Path
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def load_image(images_root, relative_path, size):
    root = Path(images_root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image escapes images-root: {relative_path}")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        pixels = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return pixels.reshape(size, size, 3).permute(2, 0, 1).float().div_(255).sub_(0.5).div_(0.5)


class CaptionDataset(Dataset):
    def __init__(self, records, images_root, tokenizer, image_size, max_tokens):
        self.records, self.images_root, self.tokenizer = records, images_root, tokenizer
        self.image_size, self.max_tokens = image_size, max_tokens
        self.encoded = []
        for r in records:
            ids = tokenizer.encode(r["caption"])
            if len(ids) + 1 > max_tokens:
                raise ValueError(f"image_id={r['image_id']}: {len(ids)+1} target tokens exceed max_text_tokens={max_tokens}; increase context explicitly")
            self.encoded.append(ids)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r, ids = self.records[i], self.encoded[i]
        return {"images": load_image(self.images_root, r["image_path"], self.image_size),
                "input_ids": torch.tensor([self.tokenizer.BOS] + ids, dtype=torch.long),
                "targets": torch.tensor(ids + [self.tokenizer.EOS], dtype=torch.long), "image_id": r["image_id"]}


def collate(examples):
    return {"images": torch.stack([e["images"] for e in examples]),
            "input_ids": torch.nn.utils.rnn.pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=0),
            "targets": torch.nn.utils.rnn.pad_sequence([e["targets"] for e in examples], batch_first=True, padding_value=0),
            "image_ids": [e["image_id"] for e in examples]}


def text_stats(dataset):
    ids = [i for row in dataset.encoded for i in row]
    return {"records": len(dataset), "tokens": len(ids), "unknown_tokens": sum(i == dataset.tokenizer.UNK for i in ids),
            "unknown_rate": sum(i == dataset.tokenizer.UNK for i in ids) / max(1, len(ids)),
            "max_target_tokens": max(len(row) + 1 for row in dataset.encoded)}
