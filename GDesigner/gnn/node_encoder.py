from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """Lightweight encoder that maps a textual query to a node embedding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        embedding_fn: Optional[Callable[[Union[str, np.ndarray]], np.ndarray]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.embedding_fn = embedding_fn
        self.projection = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Encode a pre-computed query embedding."""

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        encoded = self.projection(query_embedding)
        encoded = self.activation(encoded)

        if squeeze_back:
            encoded = encoded.squeeze(0)
        return encoded

    def encode(self, query: str) -> torch.Tensor:
        if self.embedding_fn is None:
            raise ValueError("embedding_fn is not provided; cannot encode from raw text.")
        embedding = self.embedding_fn(query)
        if isinstance(embedding, np.ndarray):
            tensor = torch.tensor(embedding, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(embedding, dtype=torch.float32)
        return self.forward(tensor)
