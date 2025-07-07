import glob

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer

from action import ActionDiscretizer
from openvla import OpenVLA

# Vision preprocessing
vision_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
    ]
)

# Language preprocessing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


class OpenVLADataset(Dataset):
    def __init__(self, vision_transform, tokenizer, discretizer, max_len=64):
        self.vision_transform = vision_transform
        self.tokenizer = tokenizer
        self.discretizer = discretizer
        self.max_len = max_len
        self.file_paths = glob.glob("data/*.h5")

        self.sample_index = []
        self.file_handles = {}

        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, "r") as f:
                num_samples = len(f["actions"])
                self.sample_index.extend(
                    [(file_idx, sample_idx) for sample_idx in range(num_samples)]
                )

    def _get_file_handle(self, file_idx):
        """Lazy file handle caching"""
        if file_idx not in self.file_handles:
            self.file_handles[file_idx] = h5py.File(self.file_paths[file_idx], "r")
        return self.file_handles[file_idx]

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_index[idx]
        f = self._get_file_handle(file_idx)

        # Load only the specific sample (not the whole array)
        image_np = f["images"][sample_idx]
        instr = f["instrs"][sample_idx]
        action_vec = f["actions"][sample_idx]

        # Convert bytes to string if needed
        if isinstance(instr, bytes):
            instr = instr.decode()

        # Vision
        image_tensor = self.vision_transform(image_np)

        # Language
        tokenized = self.tokenizer(
            instr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        lang_tokens = tokenized["input_ids"].squeeze(0)

        # Discretize action
        action_tensor = torch.tensor(action_vec, dtype=torch.float32)
        action_tokens = torch.tensor(self.discretizer(action_tensor), dtype=torch.long)

        return {
            "image": image_tensor,
            "language_tokens": lang_tokens,
            "action_tokens": action_tokens,
        }

    def __del__(self):
        # Clean up file handles
        for f in self.file_handles.values():
            f.close()


dataset = OpenVLADataset(
    vision_transform=vision_transform,
    tokenizer=tokenizer,
    discretizer=ActionDiscretizer(tokenizer=tokenizer, action_dim=7, num_bins=256),
    max_len=64,
)

# for i in range(10):
#     print(dataset[i])
openvla = OpenVLA(device="cpu")
openvla.action_discretizer = dataset.discretizer
dataloader = DataLoader(dataset, batch_size=16, num_workers=0)
optimizer = AdamW(openvla.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for sample in dataloader:
    images = sample["image"].to("cpu")
    language_tokens = sample["language_tokens"].to("cpu")
    action_tokens = sample["action_tokens"].to("cpu")

    out = openvla(images, language_tokens)
    print(out.logits.shape)

    action_token_begin_idx = dataset.discretizer.action_token_begin_idx
    logits_slice = out.logits[
        :, -7:, action_token_begin_idx : action_token_begin_idx + 256
    ]  # (B, 7, 256)
    logits_flat = logits_slice.reshape(-1, 256).contiguous()  # (B*7, 256)

    targets_relative = action_tokens - action_token_begin_idx  # Convert to 0-255 range
    targets_flat = targets_relative.reshape(-1)  # (B*7,)

    loss = F.cross_entropy(logits_flat, targets_flat)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
