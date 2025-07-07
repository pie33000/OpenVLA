import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from action import ActionDiscretizer
from vision_encoder import VisionEncoder

warnings.filterwarnings("ignore")


class OpenVLA(nn.Module):
    def __init__(
        self, dim: int = 896, device: str = "cuda", action_dim: int = 7, num_bins: int = 256
    ):
        super(OpenVLA, self).__init__()
        self.device = device
        self.dim = dim
        self.vision_encoder = VisionEncoder(dim=dim).to(device)
        self._qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-0.5B", torch_dtype="auto", device_map=device
        )
        self.language_model_embeddings = deepcopy(self._qwen.model.embed_tokens)
        # remove the embeddings layer
        self._qwen.model.embed_tokens = nn.Identity()
        self.language_model = deepcopy(self._qwen)
        self._qwen = None

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.action_discretizer = None

    def embed_text(self, text):
        if isinstance(text, str):
            tokens = self.tokenizer.encode(text)
            tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        else:
            tokens = text
            tokens = torch.tensor(tokens).to(self.device)
        return self.language_model_embeddings(tokens)

    def get_actions(self, x):
        x = x.clone()
        # Mask out everything except action tokens
        action_token_begin_idx = self.action_discretizer.action_token_begin_idx
        x[:, :, :action_token_begin_idx] = -100  # mask non-action tokens

        token_ids = torch.argmax(x, dim=-1)
        last_tokens = token_ids[:, -self.action_discretizer.action_dim :]

        # Convert token IDs to continuous actions using the discretizer
        action_tokens_np = last_tokens.cpu().numpy()
        continuous_actions = []
        for batch_idx in range(action_tokens_np.shape[0]):
            batch_actions = self.action_discretizer.decode_token_ids_to_actions(
                action_tokens_np[batch_idx]
            )
            continuous_actions.append(batch_actions)

        return torch.tensor(continuous_actions, device=x.device)

    def forward(self, image, text):
        image_embeddings = self.vision_encoder(image)
        text_embeddings = self.embed_text(text)
        x = torch.cat([image_embeddings, text_embeddings], dim=1)
        x = x.to(self.language_model.dtype)
        x = self.language_model(x)
        # action decoding and encoding
        return x


# image = torch.randn(1, 3, 224, 224).to("cuda")
# task = "pick up the red ball"
# text = f"What should the robot do to {task}? A:"
# openvla = OpenVLA(device="cuda")
# openvla.action_discretizer.fit(torch.rand(100000, 7).to("cuda"))
# out = openvla(image, text)
# actions = openvla.get_actions(out.logits)
# print(actions)
