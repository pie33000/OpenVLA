import torch


class ActionDiscretizer:
    def __init__(self, action_dim: int = 7, num_bins: int = 256):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.bin_edges = None     # shape: [action_dim, 2]
        self.bin_widths = None    # shape: [action_dim]

    @torch.no_grad()
    def fit(self, action_data: torch.Tensor):
        """
        Compute 1st to 99th percentile range for each action dimension.
        """
        assert action_data.ndim == 2 and action_data.shape[1] == self.action_dim, \
            f"Expected shape (N, {self.action_dim}), got {action_data.shape}"

        device = action_data.device
        dtype = action_data.dtype

        low = torch.quantile(action_data, 0.01, dim=0)
        high = torch.quantile(action_data, 0.99, dim=0)

        self.bin_edges = torch.stack((low, high), dim=1).to(device=device, dtype=dtype)
        self.bin_widths = (high - low) / self.num_bins

    @torch.no_grad()
    def encode(self, action: torch.Tensor) -> torch.LongTensor:
        """
        Discretize continuous actions into integer tokens ∈ [0, num_bins - 1]
        Supports shape (action_dim,) or (B, action_dim)
        """
        if action.ndim == 1:
            action = action.unsqueeze(0)  # [1, action_dim]

        low = torch.tensor(self.bin_edges[:, 0], dtype=action.dtype, device=action.device)       # [action_dim]
        width = torch.tensor(self.bin_widths, dtype=action.dtype, device=action.device)          # [action_dim]

        # [B, action_dim] - [action_dim] -> broadcasting
        tokens = ((action - low) / width).clamp(0, self.num_bins - 1)
        return tokens.floor().to(dtype=torch.long, device=action.device)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens back into continuous action values.
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)  # [1, action_dim]

        low = torch.tensor(self.bin_edges[:, 0], dtype=tokens.dtype, device=tokens.device)
        width = torch.tensor(self.bin_widths, dtype=tokens.dtype, device=tokens.device)

        return tokens * width + low