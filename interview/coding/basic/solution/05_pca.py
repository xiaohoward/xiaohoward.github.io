"""Principal component analysis via SVD

Implement PCA as a small class. fit() centers the data, computes the principal directions with an SVD of the
centered data matrix (NOT by forming the covariance and calling eigh), and stores the explained variance and its
ratio. transform() projects onto the top n_components directions; inverse_transform() maps back to the original
space (a rank-n_components reconstruction).

Signatures (class PCA):
    PCA(n_components).fit(X) -> self
    PCA.transform(X) -> (N, n_components)
    PCA.inverse_transform(Z) -> (N, D)
    attributes after fit: mean_ (D,), components_ (n_components, D), explained_variance_ (n_components,),
                          explained_variance_ratio_ (n_components,), singular_values_ (n_components,)

Constraints: numpy only. X is (N, D) float64, N > 1. Use np.linalg.svd(Xc, full_matrices=False). Variance uses
the unbiased 1/(N-1) normalization. Fix the sign of each component so that its largest-|entry| element is
positive (so results are deterministic and comparable to a reference). explained_variance_ratio_ is relative
to the TOTAL variance (sum over all D directions), not to the kept ones.

Interview budget: 20 min

Discussion follow-ups:
  - Why SVD of X rather than eigh of X^T X? (Conditioning: squaring the singular values.) Cost O(N D min(N, D)).
    When N >> D, is forming the covariance fine? When D >> N, what trick do you use?
  - Randomized SVD / power iteration for the top-k components on huge matrices; incremental PCA.
  - PCA as the linear autoencoder minimum / optimal rank-k reconstruction (Eckart-Young). Relation to whitening,
    to the latent-space power spectrum in diffusion models, and to the DCT for stationary signals.
  - What happens without centering? Why is scaling features first sometimes essential?
"""
import numpy as np

__implement__ = ["PCA.fit", "PCA.transform", "PCA.inverse_transform"]


def make_data(n=500, d=10, rank=3, noise=0.05, seed=0):
    """Given helper: data living near a random rank-`rank` affine subspace of R^d plus isotropic noise. Returns (N, D)."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    scales = np.array([3.0, 2.0, 1.0])[:rank] if rank <= 3 else rng.uniform(1, 3, size=rank)
    Z = rng.normal(size=(n, rank)) * scales
    return Z @ basis.T + rng.normal(size=d) + noise * rng.normal(size=(n, d))


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        """Fit on X (N, D) float64, N > 1, 1 <= n_components <= min(N, D).

        Steps: mean_ = X.mean(0); Xc = X - mean_; U, S, Vt = svd(Xc, full_matrices=False).
        components_ = Vt[:k] with each row's sign flipped so that its max-|.| entry is positive.
        singular_values_ = S[:k]; explained_variance_ = S[:k]^2 / (N - 1);
        explained_variance_ratio_ = explained_variance_ / (sum over ALL S^2 / (N - 1)).
        Returns self.
        """
        n = X.shape[0]
        k = self.n_components
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:k]
        signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
        signs[signs == 0] = 1.0
        self.components_ = comps * signs[:, None]
        self.singular_values_ = S[:k].copy()
        all_var = S ** 2 / (n - 1)
        self.explained_variance_ = all_var[:k]
        self.explained_variance_ratio_ = all_var[:k] / all_var.sum()
        return self

    def transform(self, X):
        """Project X (N, D) onto the components: returns (X - mean_) @ components_.T of shape (N, n_components)."""
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        """Map scores Z (N, n_components) back to data space: Z @ components_ + mean_, shape (N, D)."""
        return Z @ self.components_ + self.mean_
