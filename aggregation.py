"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


_TAIL_TOKENS = 48
_LAYER_INDICES = (16, 20, 24)


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Pick spaced late transformer layers, skipping embeddings."""
    max_layer = n_layers - 1
    return [idx for idx in _LAYER_INDICES if 1 <= idx <= max_layer]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    attention_mask = attention_mask.to(device=hidden_states.device)
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        real_positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    last_pos = int(real_positions[-1].item())
    real_hidden = hidden_states[:, real_positions, :]
    tail_hidden = hidden_states[:, real_positions[-_TAIL_TOKENS:], :]

    features: list[torch.Tensor] = []

    for layer_idx in _selected_layer_indices(hidden_states.shape[0]):
        layer = hidden_states[layer_idx]
        real_layer = real_hidden[layer_idx]
        tail_layer = tail_hidden[layer_idx]

        features.extend(
            [
                layer[last_pos],
                tail_layer.mean(dim=0),
                real_layer.mean(dim=0),
            ]
        )

    final_real = real_hidden[-1]
    final_tail = tail_hidden[-1]
    prev_tail = tail_hidden[-2] if hidden_states.shape[0] > 1 else final_tail

    # Low-dimensional statistical features help the linear probe use sequence
    # length, activation scale, and final-layer drift without enabling
    # USE_GEOMETRIC in solution.py.
    scalar_features = torch.stack(
        [
            attention_mask.float().sum(),
            final_tail.norm(dim=1).mean(),
            final_tail.norm(dim=1).std(unbiased=False),
            final_real.norm(dim=1).mean(),
            torch.nn.functional.cosine_similarity(
                final_tail.mean(dim=0), prev_tail.mean(dim=0), dim=0
            ),
            (final_tail[-1] - final_tail[0]).norm(),
        ]
    ).to(dtype=hidden_states.dtype)

    features.append(scalar_features)
    return torch.cat(features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    attention_mask = attention_mask.to(device=hidden_states.device)
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        real_positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    real_hidden = hidden_states[:, real_positions, :]
    tail_hidden = real_hidden[:, -_TAIL_TOKENS:, :]
    selected = _selected_layer_indices(hidden_states.shape[0])

    values: list[torch.Tensor] = [attention_mask.float().sum()]
    for layer_idx in selected:
        layer = real_hidden[layer_idx]
        tail = tail_hidden[layer_idx]
        values.extend(
            [
                layer.norm(dim=1).mean(),
                layer.norm(dim=1).std(unbiased=False),
                tail.norm(dim=1).mean(),
                tail.norm(dim=1).std(unbiased=False),
            ]
        )

    for lower, upper in zip(selected, selected[1:]):
        values.append(
            torch.nn.functional.cosine_similarity(
                tail_hidden[lower].mean(dim=0), tail_hidden[upper].mean(dim=0), dim=0
            )
        )

    return torch.stack(values).to(dtype=hidden_states.dtype)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
