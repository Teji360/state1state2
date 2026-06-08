import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


def sketch_to_range_basis(phi: jnp.ndarray, K: jnp.ndarray, rank: int) -> jnp.ndarray:
    """
    Halko et al. (2011) randomized range finder using the existing Phi sketch matrix.

    phi rows are random directions (same initialization as Phi_init in JaxTensorSketchStore).
    Reusing them as the random sketch Omega for SVD connects the count-min sketch to
    optimal low-rank approximation — the core theoretical bridge for the paper.

    phi: (m, d), K: (t, d), rank: r <= m
    Returns Q: (t, rank) orthonormal basis for the approximate range of K.
    """
    Y = K @ phi[:rank].T      # (t, rank) — random projection of K's row space
    Q, _ = jnp.linalg.qr(Y)  # (t, rank) — orthonormal basis
    return Q

class SketchState(NamedTuple):
    S: jnp.ndarray
    Phi: jnp.ndarray

class JaxTensorSketchStore:
    """
        Intializes a Sketch Satte store that has a depth, width and embedding dimensionality
    """
    def __init__(self, depth: int, width: int, embedding_dim: int, seed: int = 42):
        self.d = depth
        self.w = width
        self.embedding_dim = embedding_dim

        # JAX requires explicit PRNGKeys for repreductability
        key = jax.random.PRNGKey(seed)

        stddev = jnp.sqrt(2.0 / embedding_dim)
        self.Phi_init = jax.random.normal(key, (self.d * self.w, self.embedding_dim)) * stddev

    
    def init_state(self) -> SketchState:
        """Create an iniaal impy state tracking container instance"""
        S_init = jnp.zeros((self.d, self.w), dtype = jnp.float32)
        return SketchState(S=S_init, Phi=self.Phi_init)
    


    @staticmethod
    @jax.jit
    def inner_loop_update(state: SketchState, token_embedding: np.ndarry, d: int, w: int) -> SketchState:
        """
            Compiles XLA to be balzing-fast, hardware-native speeds
            Projects a token embedding and returns an updated immutable sketch
        """
        flat_projection = state.Phi @ token_embedding
        new_S = state.S + flat_projection.reshape((d, w))
        return state._replace(S=new_S)
