"""Didactic example of an autoregressive Transformer-based language model.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
- V: Vocabulary size.
"""

import haiku as hk
import jax
import jax.numpy as jnp


class CodeAZNet(hk.Module):
    def __init__(
        self,
        num_actions: int = 11,
        d_model: int = 128,
        name: str = "simple_az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.d_model = d_model

    def __call__(
        self,
        token_ids: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # token_ids: [B, T]
        B, T = token_ids.shape

        # Simple embedding layer
        embed = hk.Embed(vocab_size=50000, embed_dim=self.d_model)(
            token_ids
        )  # [B, T, D]
        x = jnp.mean(embed, axis=1)  # Simply pool to [B, D].

        # Dual-layer MLP processing
        x = hk.Linear(self.d_model)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.d_model)(x)
        x = jax.nn.relu(x)

        # Strategic Leader
        logits = hk.Linear(self.num_actions)(x)  # [B, num_actions]

        # Value Leader
        value = hk.Linear(1)(x)
        value = jnp.tanh(value)
        value = value.squeeze(-1)  # [B]

        return logits, value
