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
    @jax.jit(static_argnums=(2, 3))
    def inner_loop_update(state: SketchState, token_embedding: jnp.ndarray, d: int, w: int) -> SketchState:
        """
            Compiles XLA to be balzing-fast, hardware-native speeds
            Projects a token embedding and returns an updated immutable sketch
        """
        flat_projection = state.Phi @ token_embedding
        new_S = state.S + flat_projection.reshape((d, w))
        return state._replace(S=new_S)

    @staticmethod
    @jax.jit(static_argnums=(2, 3))
    def query(state: SketchState, q: jnp.ndarray, d: int, w: int) -> jnp.ndarray:
        """
        Count-min inner-product query for embedding q.

        S[i, j] = sum over all seen tokens x of (Phi[i*w+j] · x), so the inner
        product (Phi_i @ q) · S[i] estimates sum_t(q · x_t) for each row i.
        Taking the minimum across rows gives the count-min estimate — each row
        provides an independent unbiased overestimate, and the minimum reduces
        the variance while preserving the lower-bound guarantee.

        Returns a scalar: estimated aggregate similarity of q to the token stream.
        """
        flat_proj = state.Phi @ q                        # (d*w,)
        proj = flat_proj.reshape((d, w))                 # (d, w) — q projected per row
        row_estimates = jnp.sum(proj * state.S, axis=1)  # (d,)   — inner product per row
        return jnp.min(row_estimates)                    # count-min: minimum over rows


def sketch_cold_start(
    K: jnp.ndarray,
    V_mat: jnp.ndarray,
    store: JaxTensorSketchStore,
    sketch_state: SketchState,
    R_max: int,
    T_max: int,
) -> "CompressedKVState":
    """
    Sketch-driven cold start: the core theoretical bridge for the paper.

    The same Phi matrix that drives the count-min frequency sketch is reused
    as the Halko randomized SVD sketch (Omega), so both compression schemes
    share a single random projection.  Importance weights are derived by
    querying each token's key vector against the accumulated sketch — tokens
    whose keys align with the seen distribution score higher and bias the SVD
    to retain their singular directions.

    Requires store.d * store.w >= R_max so Phi has enough rows for the
    randomized range finder to span an R_max-dimensional subspace.
    """
    from kv_state import cold_start

    t, d = K.shape

    # Query every key vector against the sketch in one vectorised pass.
    # Each score estimates aggregate similarity of that token to the stream;
    # negatives (possible from random projection) are clamped to zero.
    importance = jax.vmap(
        lambda k: JaxTensorSketchStore.query(sketch_state, k, store.d, store.w)
    )(K)                                      # (t,)
    importance = jnp.maximum(importance, 0.0)

    return cold_start(
        K, V_mat,
        phi=store.Phi_init,
        R_max=R_max,
        T_max=T_max,
        d=d,
        importance_weights=importance,
    )
