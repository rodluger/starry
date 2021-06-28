# -*- coding: utf-8 -*-
from ._core.math import greedy_linalg as linalg
from ._core.math import greedy_math as math
from ._core.math import nadam
import numpy as np
from tqdm.auto import tqdm
from scipy.linalg import block_diag
import theano
import theano.tensor as tt


cho_factor = math.cholesky
cho_solve = linalg.cho_solve


class Solve:
    def __init__(self, map):
        self.Ny = map.Ny
        self.nt = map.nt
        self.nw = map.nw
        self.nw0 = map.nw0
        self.nc = map.nc
        self._continuum_idx = map._continuum_idx

        self.meta = {}

        if map.lazy:
            # TODO!
            raise NotImplementedError("TODO: Compile methods.")

            self._Se2i = map._Se2i.eval()
            self._S0e2i = map._S0e2i.eval()

        else:
            self.design_matrix_fixed_spectrum = lambda theta: map.design_matrix(
                theta=theta, fix_spectrum=True
            )
            self.kT0_matrix = lambda: map.ops.get_kT0_matrix(
                map._veq, map._inc
            )
            self.L1 = map.ops.L1

            self._Se2i = map._Se2i
            self._S0e2i = map._S0e2i

    def _process_inputs(
        self,
        flux,
        flux_err=None,
        theta=None,
        spatial_mean=None,
        spatial_cov=None,
        spatial_inv_cov=None,
        spectral_mean=None,
        spectral_lambda=None,
        spectral_maxiter=None,
        spectral_eps=None,
        spectral_tol=None,
        normalized=True,
        baseline=None,
        baseline_var=None,
        fix_spectrum=False,
        fix_map=False,
        logT0=None,
        logTf=None,
        nlogT=None,
        lr=None,
        niter=None,
        quiet=False,
    ):
        # --------------------------
        # ---- Process defaults ----
        # --------------------------

        if flux_err is None:
            flux_err = 1e-6
        if spatial_mean is None:
            spatial_mean = 0.0
        if spatial_cov is None and spatial_inv_cov is None:
            spatial_cov = 1e-4 * np.ones(self.Ny)
            spatial_cov[0] = 1
        if baseline_var is None:
            baseline_var = 1e-2
        if logT0 is None:
            logT0 = 12
        if logTf is None:
            logTf = 0
        if nlogT is None:
            nlogT = 50
        else:
            nlogT = max(1, nlogT)
        if lr is None:
            lr = 2e-5
        if niter is None:
            niter = 0
        if spectral_mean is None:
            spectral_mean = 1.0
        if spectral_lambda is None:
            spectral_lambda = 1e6
        if spectral_maxiter is None:
            spectral_maxiter = 100
        if spectral_eps is None:
            spectral_eps = 1e-12
        if spectral_tol is None:
            spectral_tol = 1e-8

        # ----------------------
        # ---- Check shapes ----
        # ----------------------

        # Flux must be a matrix (nt, nw)
        if self.nt == 1:
            self.flux = np.reshape(flux, (self.nt, self.nw))
        else:
            assert np.array_equal(
                np.shape(flux), np.array([self.nt, self.nw])
            ), "Invalid shape for `flux`."
            self.flux = flux

        # Flux error may be a scalar, a vector, or a matrix (nt, nw)
        flux_err = np.array(flux_err)
        if flux_err.ndim == 0:
            self.flux_err = flux_err
        elif flux_err.ndim == 1:
            self.flux_err = flux_err.reshape(-1, 1) * np.ones((1, self.nw))
        else:
            if self.nt == 1:
                self.flux_err = np.reshape(flux_err, (self.nt, self.nw))
            else:
                assert np.array_equal(
                    np.shape(flux_err), np.array([self.nt, self.nw])
                ), "Invalid shape for `flux_err`."
                self.flux_err = flux_err

        # Spatial mean may be a scalar, a vector (Ny), or a list of those
        # Reshape it to a matrix of shape (Ny, nc)
        if type(spatial_mean) not in (list, tuple):
            # Use the same mean for all components
            spatial_mean = [spatial_mean for n in range(self.nc)]
        else:
            # Check that we have one mean per component
            assert len(spatial_mean) == self.nc
        for n in range(self.nc):
            spatial_mean[n] = np.array(spatial_mean[n])
            assert spatial_mean[n].ndim < 2
            spatial_mean[n] = np.reshape(
                spatial_mean[n] * np.ones(self.Ny), (-1, 1)
            )
        self.spatial_mean = np.concatenate(spatial_mean, axis=-1)

        # Spatial (inv) cov may be a scalar, a vector, a matrix (Ny, Ny),
        # or a list of those. Invert it if needed and reshape to a matrix of
        # shape (Ny, nc) (inverse variances) or a tensor of shape
        # (Ny, Ny, nc) (nc separate inverse covariance matrices)
        if spatial_cov is not None:

            # User provided the *covariance*

            if type(spatial_cov) not in (list, tuple):
                # Use the same covariance for all components
                spatial_cov = [spatial_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spatial_cov) == self.nc
            spatial_inv_cov = [None for n in range(self.nc)]

            for n in range(self.nc):
                spatial_cov[n] = np.array(spatial_cov[n])
                assert spatial_cov[n].ndim == spatial_cov[0].ndim
                if spatial_cov[n].ndim < 2:
                    spatial_inv_cov[n] = np.reshape(
                        np.ones(self.Ny) / spatial_cov[n], (-1, 1)
                    )
                else:
                    cho = cho_factor(spatial_cov[n])
                    inv = cho_solve(cho, np.eye(self.Ny))
                    spatial_inv_cov[n] = np.reshape(inv, (self.Ny, self.Ny, 1))
        else:

            # User provided the *inverse covariance*

            if type(spatial_inv_cov) not in (list, tuple):
                # Use the same covariance for all components
                spatial_inv_cov = [spatial_inv_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spatial_inv_cov) == self.nc

            for n in range(self.nc):
                spatial_inv_cov[n] = np.cast(spatial_inv_cov[n])
                assert spatial_inv_cov[n].ndim == spatial_inv_cov[0].ndim
                if spatial_inv_cov[n].ndim < 2:
                    spatial_inv_cov[n] = np.reshape(
                        np.ones(self.Ny) * spatial_inv_cov[n], (-1, 1)
                    )
                else:
                    spatial_inv_cov[n] = np.reshape(
                        spatial_inv_cov[n], (self.Ny, self.Ny, 1)
                    )

        # Tensor of nc inverse variance vectors or covariance matrices
        self.spatial_inv_cov = np.concatenate(spatial_inv_cov, axis=-1)

        # Baseline must be a vector (nt,)
        if baseline is not None:
            assert np.array_equal(
                np.shape(baseline), np.array([self.nt])
            ), "Invalid shape for `baseline`."
            self.baseline = baseline
        else:
            self.baseline = None

        # Spectral mean must be a scalar, a vector, or a matrix (nc, nw0)
        spectral_mean = np.array(spectral_mean)
        if spectral_mean.ndim == 0:
            self.spectral_mean = spectral_mean * np.ones((self.nc, self.nw0))
        elif spectral_mean.ndim == 1:
            assert np.array_equal(
                np.shape(spectral_mean), np.array([self.nc])
            ), "Invalid shape for `spectral_mean`."
            self.spectral_mean = np.reshape(
                spectral_mean, (self.nc, 1)
            ) * np.ones((self.nc, self.nw0))
        else:
            assert np.array_equal(
                np.shape(spectral_mean), np.array([self.nc, self.nw0])
            ), "Invalid shape for `spectral_mean`."
            self.spectral_mean = spectral_mean

        # Tempering schedule
        if nlogT == 1:
            self.T = np.array([10 ** logTf])
        else:
            self.T = np.logspace(logT0, logTf, nlogT)

        # Ingest the remaining params
        self.theta = theta
        self.spectral_lambda = spectral_lambda
        self.spectral_maxiter = spectral_maxiter
        self.spectral_eps = spectral_eps
        self.spectral_tol = spectral_tol
        self.normalized = normalized
        self.baseline_var = baseline_var
        self.fix_spectrum = fix_spectrum
        self.fix_map = fix_map
        self.lr = lr
        self.niter = niter
        self.quiet = quiet

    def solve_for_map_linear(self, D=None, C=None):
        """
        Solve for `y` linearly, given a baseline or unnormalized data.

        """

        # Get the design matrix conditioned on the current spectrum
        if D is None:
            D = self.design_matrix_fixed_spectrum(self.theta)

        # Get the design matrix for the continuum normalization
        if C is None:
            C = np.reshape(D, [self.nt, self.nw, -1])[
                :, self._continuum_idx, :
            ]

        # Reshape the priors
        mu = np.reshape(np.transpose(self.spatial_mean), (-1))
        if self.spatial_inv_cov.ndim == 2:
            invL = np.reshape(np.transpose(self.spatial_inv_cov), (-1))
        else:
            invL = block_diag(
                *[self.spatial_inv_cov[:, :, n] for n in range(self.nc)]
            )

        # Ensure the flux error is a vector
        if self.flux_err.ndim == 0:
            flux_err = self.flux_err * np.ones((self.nt, self.nw))
        else:
            flux_err = self.flux_err

        # Get the factorized data covariance
        if self.baseline is not None:

            # De-normalize the data w/ the given baseline
            flux = self.flux * np.reshape(self.baseline, (self.nt, 1))
            flux_err *= np.reshape(self.baseline, (self.nt, 1))
            cho_C = np.reshape(flux_err, (-1,))

        else:

            # Easy!
            flux = self.flux
            cho_C = np.reshape(flux_err, (-1,))

        # Unroll the data into a vector
        flux = np.reshape(flux, (-1,))

        # Solve the L2 problem
        mean, cho_cov = linalg.solve(D, flux, cho_C, mu, invL)
        y = np.transpose(np.reshape(mean, (self.nc, self.Ny)))
        self.meta["y_lin"] = y

        return y, cho_cov

    def solve_for_map_tempered(self, D=None, C=None):
        """
        Solve for `y` with an iterative linear, tempered solver.

        """

        # Get the design matrix conditioned on the current spectrum
        if D is None:
            D = self.design_matrix_fixed_spectrum(self.theta)

        # Get the design matrix for the continuum normalization
        if C is None:
            C = np.reshape(D, [self.nt, self.nw, -1])[
                :, self._continuum_idx, :
            ]

        # Unroll the data into a vector
        flux = np.reshape(self.flux, (-1,))

        # Reshape the priors
        mu = np.reshape(np.transpose(self.spatial_mean), (-1))
        if self.spatial_inv_cov.ndim == 2:
            invL = np.reshape(np.transpose(self.spatial_inv_cov), (-1))
        else:
            invL = block_diag(
                *[self.spatial_inv_cov[:, :, n] for n in range(self.nc)]
            )

        # Tempering to find the baseline
        baseline = np.ones(self.nt * self.nw)
        for i in tqdm(range(len(self.T)), disable=self.quiet):

            # Marginalize over the unknown baseline at each epoch
            # and compute the factorized data covariance
            if self.flux_err.ndim == 0:
                cho_C = cho_factor(
                    self.T[i] * self.flux_err ** 2 * np.eye(self.nw)
                    + self.baseline_var
                )
                cho_C = block_diag(*[cho_C for n in range(self.nt)])
            elif self.flux_err.ndim == 2:
                cho_C = block_diag(
                    *[
                        cho_factor(
                            self.T[i] * np.diag(self.flux_err[n] ** 2)
                            + self.baseline_var
                        )
                        for n in range(self.nt)
                    ]
                )
            else:
                raise ValueError("Invalid shape for `flux_err`.")

            # Solve the L2 problem for the Ylm coeffs
            y_flat, cho_cov = linalg.solve(D, flux * baseline, cho_C, mu, invL)
            y = np.transpose(np.reshape(y_flat, (self.nc, self.Ny)))

            # Refine the baseline estimate
            baseline = np.dot(C, y_flat)
            baseline = np.repeat(baseline, self.nw, axis=0)
            baseline = np.reshape(baseline, (self.nt * self.nw,))

        self.meta["y_temp"] = y

        return y, cho_cov

    def solve_for_map_nadam(self, y, D=None, C=None):
        """
        Solve for `y` with gradient descent given a guess.

        """

        # Get the design matrix conditioned on the current spectrum
        if D is None:
            D = self.design_matrix_fixed_spectrum(self.theta)

        # Get the design matrix for the continuum normalization
        if C is None:
            C = np.reshape(D, [self.nt, self.nw, -1])[
                :, self._continuum_idx, :
            ]

        # Unroll the data into a vector
        flux = np.reshape(self.flux, (-1,))
        flux_err = np.reshape(
            self.flux_err * np.ones((self.nt, self.nw)), (-1,)
        )
        if self.flux_err.ndim == 0:
            cho_C = cho_factor(
                self.flux_err ** 2 * np.eye(self.nw) + self.baseline_var
            )
            cho_C = block_diag(*[cho_C for n in range(self.nt)])
        elif self.flux_err.ndim == 2:
            cho_C = block_diag(
                *[
                    cho_factor(
                        np.diag(self.flux_err[n] ** 2) + self.baseline_var
                    )
                    for n in range(self.nt)
                ]
            )
        else:
            raise ValueError("Invalid shape for `flux_err`.")

        # Reshape the priors
        mu = np.reshape(np.transpose(self.spatial_mean), (-1))
        if self.spatial_inv_cov.ndim == 2:
            invL = np.reshape(np.transpose(self.spatial_inv_cov), (-1))
        else:
            invL = block_diag(
                *[self.spatial_inv_cov[:, :, n] for n in range(self.nc)]
            )

        # Compute the exact model w/ normalization
        y_ = theano.shared(y)
        y_flat_ = tt.reshape(tt.transpose(y_), (-1,))
        baseline_ = tt.dot(C, y_flat_)
        b_ = tt.repeat(baseline_, self.nw, axis=0)
        b_ = tt.reshape(b_, (self.nt * self.nw,))
        model_ = tt.dot(D, y_flat_)
        model_ /= b_

        # The loss is -0.5 * lnL (up to a constant)
        loss_ = tt.sum((flux - model_) ** 2 / tt.reshape(flux_err, (-1,)) ** 2)
        if invL.ndim == 1:
            loss_ += tt.sum(y_flat_ ** 2 * invL)
        else:
            loss_ += tt.dot(
                tt.dot(tt.reshape(y_flat_, (1, -1)), invL),
                tt.reshape(y_flat_, (-1, 1)),
            )[0, 0]

        # Compile the NAdam optimizer and iterate
        best_loss = np.inf
        best_y = np.array(y)
        best_baseline = np.dot(C, y.T.reshape(-1))
        loss = np.zeros(self.niter)
        upd_ = nadam(loss_, [y_], lr=self.lr)
        train = theano.function([], [y_, baseline_, loss_], updates=upd_)
        for n in tqdm(range(self.niter), disable=self.quiet):
            y, baseline, loss[n] = train()
            if loss[n] < best_loss:
                best_loss = loss[n]
                best_y = y
                best_baseline = baseline
        y = best_y
        baseline = best_baseline
        self.meta["loss"] = loss
        self.meta["y_nadam"] = y

        # Final step: using our refined baseline estimate,
        # re-compute the MAP solution to obtain an estimate
        # of the posterior variance
        b = np.repeat(baseline, self.nw, axis=0)
        b = np.reshape(b, (self.nt * self.nw,))
        y_lin, cho_cov = linalg.solve(D, flux * b, cho_C, mu, invL)
        y_lin = np.transpose(np.reshape(y, (self.nc, self.Ny)))
        self.meta["y_lin"] = y_lin

        return y, cho_cov

    def solve(self, flux, **kwargs):
        """
        Solve the linear problem for the spatial and/or spectral map
        given a spectral timeseries.

        .. warning::

            This method is still being developed!

        """
        # Parse the inputs
        self._process_inputs(flux, **kwargs)

        # Metadata for output
        self.meta = {}

        if self.fix_spectrum:

            # The spectrum is fixed. We are solving the for spatial map.

            if (not self.normalized) or (self.baseline is not None):

                # The problem is exactly linear!
                y, cho_cov = self.solve_for_map_linear()

            else:

                # The problem is linear conditioned on a baseline
                # guess, which we refine iteratively
                y, cho_cov = self.solve_for_map_tempered()

                # Refine the solution with gradient descent
                if self.niter > 0:
                    y, cho_cov = self.solve_for_map_nadam(y)

            # Return the MAP and factorized posterior covariance
            return dict(y=y, cho_cov=cho_cov, **self.meta)

        elif self.fix_map:

            # TODO
            raise NotImplementedError("Not yet implemented.")

        else:

            # We need to solve for both the map and the spectrum

            # First, deconvolve the flux to obtain a guess for the spectrum
            # We subtract the convolved mean spectrum and add the deconvolved
            # residuals to each of the component spectral means to obtain
            # a rough guess for all components. This is by no means optimal
            # when nc > 1, but it's a decent starting guess.
            A = self.kT0_matrix()
            f = self._Se2i.dot(np.mean(self.flux, axis=0))
            mu = self._S0e2i.dot(self.spectral_mean.T)
            f -= np.mean(np.dot(A, mu), axis=1)
            spectrum = mu.T + self.L1(
                A,
                f,
                np.array(self.spectral_lambda),
                np.array(np.mean(self.flux_err)),
                self.spectral_maxiter,
                np.array(self.spectral_eps),
                np.array(self.spectral_tol),
            )

            # TODO
            raise NotImplementedError("Not yet implemented.")
