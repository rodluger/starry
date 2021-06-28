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
        # Dimensions and indices
        self.Ny = map.Ny
        self.nt = map.nt
        self.nw = map.nw
        self.nw_ = map.nw_
        self.nw0 = map.nw0
        self.nw0_ = map.nw0_
        self.nc = map.nc
        self.continuum_idx = map._continuum_idx

        # Methods and matrices
        if map.lazy:

            # TODO!
            raise NotImplementedError("TODO: Compile methods here.")

            # Interpolation matrices
            self.Se2i = map._Se2i.eval()
            self.S0e2i = map._S0e2i.eval()

        else:

            # Design matrix conditioned on current spectrum
            def _get_S():
                map._spectrum = self.spectrum_
                return map.design_matrix(theta=self.theta, fix_spectrum=True)

            self._get_S = _get_S

            #
            self.dotMT = lambda x: map.dot(x, fix_map=True, transpose=True)

            # Line broadening matrix
            self._get_KT0 = lambda: map.ops.get_kT0_matrix(map._veq, map._inc)

            # LASSO solver
            self.L1 = map.ops.L1

            # Interpolation matrices
            self.Se2i = map._Se2i
            self.S0e2i = map._S0e2i

        self.reset()

    def reset(self):
        # The spectral map
        self.spectrum_ = None
        self.y = None

        # Design matrices
        self._S = None
        self._C = None
        self._KT0 = None

        # Solution metadata
        self.meta = {}

    @property
    def KT0(self):
        """
        Line broadening matrix.

        """
        if self._KT0 is None:
            self._KT0 = self._get_KT0()
        return self._KT0

    @property
    def S(self):
        """
        Design matrix conditioned on the current spectrum.

        """
        if self._S is None:
            self._S = self._get_S()
        return self._S

    @property
    def C(self):
        """
        Continuum matrix conditioned on the current spectrum and map.

        """
        if self._C is None:
            self._C = np.reshape(self.S, [self.nt, self.nw, -1])[
                :, self.continuum_idx, :
            ]
        return self._C

    def process_inputs(
        self,
        flux,
        flux_err=None,
        theta=None,
        spatial_mean=None,
        spatial_cov=None,
        spatial_inv_cov=None,
        spectral_mean=None,
        spectral_cov=None,
        spectral_inv_cov=None,
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
        if spectral_cov is None and spectral_inv_cov is None:
            spectral_cov = 1e-4
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

        # Spectral (inv) cov may be a scalar, a vector, a matrix (nw0, nw0),
        # or a list of those. Invert it if needed and reshape to a matrix of
        # shape (nc, nw0) (inverse variances) or a tensor of shape
        # (nc, nw0, nw0) (nc separate inverse covariance matrices)
        if spectral_cov is not None:

            # User provided the *covariance*

            if type(spectral_cov) not in (list, tuple):
                # Use the same covariance for all components
                spectral_cov = [spectral_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spectral_cov) == self.nc
            spectral_inv_cov = [None for n in range(self.nc)]

            for n in range(self.nc):
                spectral_cov[n] = np.array(spectral_cov[n])
                assert spectral_cov[n].ndim == spectral_cov[0].ndim
                if spectral_cov[n].ndim < 2:
                    spectral_inv_cov[n] = np.reshape(
                        np.ones(self.nw0) / spectral_cov[n], (1, -1)
                    )
                else:
                    cho = cho_factor(spectral_cov[n])
                    inv = cho_solve(cho, np.eye(self.nw0))
                    spectral_inv_cov[n] = np.reshape(
                        inv, (1, self.nw0, self.nw0)
                    )
        else:

            # User provided the *inverse covariance*

            if type(spectral_inv_cov) not in (list, tuple):
                # Use the same covariance for all components
                spectral_inv_cov = [spectral_inv_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spectral_inv_cov) == self.nc

            for n in range(self.nc):
                spectral_inv_cov[n] = np.cast(spectral_inv_cov[n])
                assert spectral_inv_cov[n].ndim == spectral_inv_cov[0].ndim
                if spectral_inv_cov[n].ndim < 2:
                    spectral_inv_cov[n] = np.reshape(
                        np.ones(self.nw0) * spectral_inv_cov[n], (1, -1)
                    )
                else:
                    spectral_inv_cov[n] = np.reshape(
                        spectral_inv_cov[n], (1, self.nw0, self.nw0)
                    )

        # Tensor of nc inverse variance vectors or covariance matrices
        self.spectral_inv_cov = np.concatenate(spectral_inv_cov, axis=0)

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

    def solve_for_map_linear(self):
        """
        Solve for `y` linearly, given a baseline or unnormalized data.

        """
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
            if self.flux_err.ndim == 0:
                flux_err = (
                    self.flux_err
                    * np.ones((1, self.nw))
                    * np.reshape(self.baseline, (self.nt, 1))
                )
            else:
                flux_err = self.flux_err * np.reshape(
                    self.baseline, (self.nt, 1)
                )
            cho_C = np.reshape(flux_err, (-1,))

        else:

            # Easy!
            flux = self.flux
            cho_C = np.reshape(flux_err, (-1,))

        # Unroll the data into a vector
        flux = np.reshape(flux, (-1,))

        # Solve the L2 problem
        mean, cho_ycov = linalg.solve(self.S, flux, cho_C, mu, invL)
        y = np.transpose(np.reshape(mean, (self.nc, self.Ny)))
        self.meta["y_lin"] = y

        # Store
        self.y = y
        self.cho_ycov = cho_ycov

    def solve_for_map_tempered(self):
        """
        Solve for `y` with an iterative linear, tempered solver.

        """
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
            y_flat, cho_ycov = linalg.solve(
                self.S, flux * baseline, cho_C, mu, invL
            )
            y = np.transpose(np.reshape(y_flat, (self.nc, self.Ny)))

            # Refine the baseline estimate
            baseline = np.dot(self.C, y_flat)
            baseline = np.repeat(baseline, self.nw, axis=0)
            baseline = np.reshape(baseline, (self.nt * self.nw,))

        self.meta["y_temp"] = y

        # Store
        self.y = y
        self.cho_ycov = cho_ycov

    def solve_for_map_nadam(self):
        """
        Solve for `y` with gradient descent given a guess.

        """
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
        y = np.array(self.y)
        y_ = theano.shared(y)
        y_flat_ = tt.reshape(tt.transpose(y_), (-1,))
        baseline_ = tt.dot(self.C, y_flat_)
        b_ = tt.repeat(baseline_, self.nw, axis=0)
        b_ = tt.reshape(b_, (self.nt * self.nw,))
        model_ = tt.dot(self.S, y_flat_)
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
        best_baseline = np.dot(self.C, y.T.reshape(-1))
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
        y_lin, cho_ycov = linalg.solve(self.S, flux * b, cho_C, mu, invL)
        y_lin = np.transpose(np.reshape(y, (self.nc, self.Ny)))
        self.meta["y_lin"] = y_lin

        # Store
        self.y = y
        self.cho_ycov = cho_ycov

    def solve_for_spectrum(self):
        """
        Solve for `spectrum_` conditioned on the current map.

        """
        # Unroll the data into a vector
        if self.normalized:

            # Un-normalize the data with the current baseline
            baseline = np.dot(self.C, self.y.T.reshape(-1))
            baseline = np.reshape(baseline, (self.nt, 1))
            flux = self.flux * baseline
            if self.flux_err.ndim == 0:
                flux_err = self.flux_err * np.ones((1, self.nw)) * baseline
            else:
                flux_err = self.flux_err * baseline
            CInv = np.diag(1.0 / flux_err.reshape(self.nt * self.nw) ** 2)

        else:

            flux = self.flux
            if self.flux_err.ndim == 0:
                CInv = (1.0 / self.flux_err ** 2) * np.eye(self.nt * self.nw)
            else:
                CInv = np.diag(
                    1.0 / self.flux_err.reshape(self.nt * self.nw) ** 2
                )

        # Unroll the data into a vector
        flux = np.reshape(flux, (-1,))

        # Reshape the priors and interpolate them onto the internal grid
        mu = self.S0e2i.dot(self.spectral_mean.T).T.reshape(-1)
        if self.spectral_inv_cov.ndim == 2:
            invL = self.S0e2i.dot(self.spectral_inv_cov.T).T.reshape(-1)
            invLmu = invL * mu
        else:
            invL = block_diag(
                *[self.S0e2i.dot(inv) for inv in self.spectral_inv_cov]
            )
            invLmu = np.dot(invL, mu)

        # Compute M^T C^-1 M
        # TODO: We can probably do this with a single convolution...
        KInv = self.dotMT(np.transpose(self.dotMT(CInv)))
        if invL.ndim == 1:
            KInv[np.diag_indices_from(KInv)] += invL
        else:
            KInv += invL

        # Solve the L2 problem
        choK = cho_factor(KInv)
        spectrum_ = cho_solve(
            choK, self.dotMT(np.dot(CInv, flux)).reshape(-1) + invLmu
        )

        # Store
        self.spectrum_ = np.reshape(spectrum_, (self.nc, self.nw0_))
        self.cho_scov = choK

    def solve(self, flux, y, spectrum_, **kwargs):
        """
        Solve the linear problem for the spatial and/or spectral map
        given a spectral timeseries.

        .. warning::

            This method is still being developed!

        """
        # Start fresh
        self.reset()

        # Parse the inputs
        self.process_inputs(flux, **kwargs)
        self.y = y
        self.cho_ycov = None
        self.spectrum_ = spectrum_
        self.cho_scov = None

        if self.fix_spectrum:

            # The spectrum is fixed. We are solving for the spatial map.

            if (not self.normalized) or (self.baseline is not None):

                # The problem is exactly linear!
                self.solve_for_map_linear()

            else:

                # The problem is linear conditioned on a baseline
                # guess, which we refine iteratively
                self.solve_for_map_tempered()

                # Refine the solution with gradient descent
                if self.niter > 0:
                    self.solve_for_map_nadam()

        elif self.fix_map:

            # The problem is exactly linear!
            self.solve_for_spectrum()

        else:

            # We need to solve for both the map and the spectrum

            # First, deconvolve the flux to obtain a guess for the spectrum
            # We subtract the convolved mean spectrum and add the deconvolved
            # residuals to each of the component spectral means to obtain
            # a rough guess for all components. This is by no means optimal
            # when nc > 1, but it's a decent starting guess.
            f = self.Se2i.dot(np.mean(self.flux, axis=0))
            f /= f[self.continuum_idx]
            mu = self.S0e2i.dot(self.spectral_mean.T)
            f -= np.mean(np.dot(self.KT0, mu), axis=1)
            self.spectrum_ = mu.T + self.L1(
                self.KT0,
                f,
                np.array(self.spectral_lambda),
                np.array(np.mean(self.flux_err)),
                self.spectral_maxiter,
                np.array(self.spectral_eps),
                np.array(self.spectral_tol),
            )

            # TODO
            raise NotImplementedError("Not yet implemented.")

        # Update the metadata with the results and return everything
        self.meta["y"] = self.y
        self.meta["cho_ycov"] = self.cho_ycov
        self.meta["spectrum_"] = self.spectrum_
        self.meta["cho_scov"] = self.cho_scov
        return self.meta
