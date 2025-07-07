from typing import List, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ActionDiscretizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        action_dim: int = 7,
        num_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, self.num_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        # Action tokens will be the last num_bins tokens in vocab
        self.action_token_begin_idx = self.tokenizer.vocab_size - self.num_bins

    def __call__(self, action: np.ndarray) -> List[int]:
        """Convert continuous actions to token IDs"""
        action = np.clip(action, self.min_action, self.max_action)
        discretized_action = np.digitize(action, self.bins)

        # Clamp to valid range [1, num_bins] then map to token IDs
        discretized_action = np.clip(discretized_action, 1, self.num_bins)
        token_ids = self.action_token_begin_idx + discretized_action - 1

        return token_ids.tolist()

    def decode_token_ids_to_actions(self, action_token_ids: Union[List, np.ndarray]) -> np.ndarray:
        """Convert token IDs back to continuous actions"""
        if isinstance(action_token_ids, list):
            action_token_ids = np.array(action_token_ids)

        # Convert token IDs back to discretized actions
        discretized_actions = action_token_ids - self.action_token_begin_idx
        discretized_actions = np.clip(discretized_actions, 0, self.num_bins - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.num_bins


# Test the implementation
if __name__ == "__main__":
    action_discretizer = ActionDiscretizer(
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B"),
        action_dim=7,
        num_bins=256,
        min_action=-1.0,
        max_action=1.0,
    )

    test_action = np.array([-1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    print(f"Original action: {test_action}")

    # Encode
    token_ids = action_discretizer(test_action)
    print(f"Token IDs: {token_ids}")
    print(f"Token ID range: {min(token_ids)} - {max(token_ids)}")

    # Decode
    decoded_action = action_discretizer.decode_token_ids_to_actions(token_ids)
    print(f"Decoded action: {decoded_action}")
    print(f"Round-trip error: {np.abs(test_action - decoded_action).max()}")
