import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
import io
from transformers import AutoTokenizer
from action import ActionDiscretizer
from openvla import OpenVLA
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F

# Vision preprocessing
vision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1]
])

# Language preprocessing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


class OpenVLADataset(Dataset):
    def __init__(self, vision_transform, tokenizer, discretizer=None, max_len=64):
        self.vision_transform = vision_transform
        self.tokenizer = tokenizer
        self.discretizer = discretizer
        self.max_len = max_len
        self.samples = np.load("data.npy", allow_pickle=True)
        self.raw_actions = np.load("raw_actions.npy", allow_pickle=True)

        # Fit discretizer once on all action data if provided
        if self.discretizer is not None and self.discretizer.bin_edges is None:
            raw_tensor = torch.tensor(np.stack(self.raw_actions), dtype=torch.float32)
            self.discretizer.fit(raw_tensor)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_np, instr, action_vec = self.samples[idx]

        # Vision
        image_tensor = self.vision_transform(image_np)

        # Language
        tokenized = self.tokenizer(
            instr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )
        lang_tokens = tokenized["input_ids"].squeeze(0)

        # Discretize action
        action_tensor = torch.tensor(action_vec, dtype=torch.float32)
        action_tokens = self.discretizer.encode(action_tensor).squeeze(0)

        return {
            "image": image_tensor,
            "language_tokens": lang_tokens,
            "action_tokens": action_tokens
        }


class OpenVLAIterable(IterableDataset):
    def __init__(self, data_path, vision_transform, tokenizer,
                 discretizer=None, max_len=64):
        self.data_path = data_path
        self.vision_transform = vision_transform
        self.tokenizer = tokenizer
        self.discretizer = discretizer
        self.max_len = max_len

        # (optional) fit discretizer once by streaming through the file
        if self.discretizer is not None and self.discretizer.bin_edges is None:
            self._fit_discretizer()

    def _fit_discretizer(self):
        actions = []
        for chunk in iter_chunks(self.data_path):
            for _, _, act in chunk:
                actions.append(act)
        raw = torch.tensor(np.stack(actions), dtype=torch.float32)
        self.discretizer.fit(raw)

    def __iter__(self):
        for chunk in iter_chunks(self.data_path):
            for image_np, instr, action_vec in chunk:
                image_tensor = self.vision_transform(image_np)
                lang_tokens = self.tokenizer(
                    instr,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len
                )["input_ids"].squeeze(0)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32)
                action_tokens = self.discretizer.encode(action_tensor).squeeze(0)
                yield {
                    "image": image_tensor,
                    "language_tokens": lang_tokens,
                    "action_tokens": action_tokens
                }


def iter_chunks(path, *, allow_pickle=True):
    """Yield one chunk at a time from an .npy file that was written in chunks."""
    with open(path, "rb") as f:
        while True:
            try:
                chunk = np.load(f, allow_pickle=allow_pickle)
            except ValueError:          # EOF reached
                break
            yield chunk                # `chunk` is whatever you saved earlier


dataset = OpenVLADataset(
    vision_transform=vision_transform,
    tokenizer=tokenizer,
    discretizer=ActionDiscretizer(action_dim=7, num_bins=256),
    max_len=64
)
openvla = OpenVLA(device="cuda")
openvla.action_discretizer = dataset.discretizer
dataloader = DataLoader(dataset, batch_size=16, num_workers=4)
optimizer = AdamW(openvla.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for sample in dataloader:
    images = sample['image'].to("cuda")
    language_tokens = sample['language_tokens'].to("cuda")
    action_tokens = sample['action_tokens'].to("cuda")

    out = openvla(images, language_tokens)
    logits_slice = out.logits[:, -7:, -256:]        # (B, 7, 256)
    logits_flat  = logits_slice.reshape(-1, 256).contiguous()   # (B*7, 256)
    targets_flat = action_tokens.reshape(-1)                   # (B*7,)
    loss = F.cross_entropy(logits_flat, targets_flat)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
