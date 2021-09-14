# -*- coding: utf-8 -*-
from ._core.math import greedy_linalg, lazy_linalg
from ._core.math import greedy_math, lazy_math
from ._core.math import nadam
from .compat import theano, tt, ts
import numpy as np
from tqdm.auto import tqdm
from scipy.linalg import block_diag


# Cholesky ops
cho_factor = greedy_math.cholesky
cho_solve = greedy_linalg.cho_solve
tt_cho_factor = lazy_math.cholesky
tt_cho_solve = lazy_linalg.cho_solve


class Solve:
    def __init__(self, map):
        # Dimensions and indices
        self.Ny = map.Ny
        self.nt = map.nt
        self.nw = map.nw
        self.nw_ = map.nw_
        self.nw0 = map.nw0
        self.nw0_ = map.nw0_
        self.wav = map.wav
        self.wav0 = map.wav0
        self.wav_ = map.wav_
        self.wav0_ = map.wav0_
        self.nc = map.nc
        self.interp = map._interp
        self.continuum_idx = map._continuum_idx

        # Methods and matrices
        if map.lazy:

            # We need to explicitly compile all tensor functions.
            # TODO: Untested!

            # Temporarily disable test values
            compute_test_value = theano.config.compute_test_value
            theano.config.compute_test_value = "off"

            # Dummy variables for compiling
            veq = tt.dscalar()
            inc = tt.dscalar()
            spectrum_ = tt.dmatrix()
            theta = tt.dvector()
            x = tt.dmatrix()
            y = tt.dmatrix()
            u = tt.dvector()
            ATA = tt.dmatrix()
            ATy = tt.dvector()
            lam = tt.dscalar()
            maxiter = tt.iscalar()
            eps = tt.dscalar()
            tol = tt.dscalar()

            # Design matrix conditioned on current spectrum
            f = map.ops.get_D_fixed_spectrum(inc, theta, veq, u, spectrum_)
            if map._interp:
                f = ts.dot(map._Si2eBlk, f)
            _get_S = theano.function(
                [inc, theta, veq, u, spectrum_], f, on_unused_input="ignore"
            )
            self._get_S = lambda: _get_S(
                self.inc, self.theta, self.veq, self.u, self.spectrum_
            )

            # Design matrix dot product conditioned on current map
            f = map.ops.dot_design_matrix_fixed_map_into(
                inc, theta, veq, u, y, x
            )
            if map._interp:
                f = ts.dot(map._Si2eBlk, f)
            _dotM = theano.function(
                [inc, theta, veq, u, y, x], f, on_unused_input="ignore"
            )
            self.dotM = lambda x: _dotM(
                self.inc, self.theta, self.veq, self.u, self.y, x
            )

            # Transpose of the the above op
            f = map.ops.dot_design_matrix_fixed_map_transpose_into(
                inc, theta, veq, u, y, x
            )
            if map._interp:
                f = ts.dot(map._Si2eBlk, f)
            _dotMT = theano.function(
                [inc, theta, veq, u, y, x], f, on_unused_input="ignore"
            )
            self.dotMT = lambda x: _dotMT(
                self.inc, self.theta, self.veq, self.u, self.y, x
            )

            # Line broadening matrix
            f = map.ops.get_kT0_matrix(veq, inc)
            _get_KT0 = theano.function([veq, inc], f)
            self._get_KT0 = lambda: _get_KT0(self.veq, self.inc)

            # LASSO solver
            self.L1 = theano.function(
                [ATA, ATy, lam, maxiter, eps, tol],
                map.ops.L1(ATA, ATy, lam, maxiter, eps, tol),
            )

            # Interpolation matrices
            if map._interp:
                self.Se2i = map._Se2i.eval()
                self.S0e2i = map._S0e2i.eval()
                self.Si2eTr = map._Si2eTr.eval()
            else:
                self.Se2i = np.eye(map.nw)
                self.S0e2i = np.eye(map.nw0)
                self.Si2eTr = np.eye(map.nw)

            # Restore test value config
            theano.config.compute_test_value = compute_test_value

        else:

            # Design matrix conditioned on current spectrum
            def _get_S():
                map._spectrum = self.spectrum_
                return map.design_matrix(
                    theta=self.theta / map._angle_factor, fix_spectrum=True
                )

            self._get_S = _get_S

            # Design matrix dot product conditioned on current map
            def _dotM(x):
                map._y = self.y
                return map.dot(
                    x,
                    theta=self.theta / map._angle_factor,
                    fix_map=True,
                    transpose=False,
                )

            def _dotMT(x):
                map._y = self.y
                return map.dot(
                    x,
                    theta=self.theta / map._angle_factor,
                    fix_map=True,
                    transpose=True,
                )

            self.dotM = _dotM
            self.dotMT = _dotMT

            # Line broadening matrix
            self._get_KT0 = lambda: map.ops.get_kT0_matrix(self.veq, self.inc)

            # LASSO solver
            self.L1 = map.ops.L1

            # Interpolation matrices
            if map._interp:
                self.Se2i = map._Se2i
                self.S0e2i = map._S0e2i
                self.Si2eTr = map._Si2eTr
            else:
                self.Se2i = np.eye(map.nw)
                self.S0e2i = np.eye(map.nw0)
                self.Si2eTr = np.eye(map.nw)

        # This method is only used in lazy mode, so we
        # don't need to compile it
        self.get_flux_from_dotconv = map.ops.get_flux_from_dotconv

        #
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
        spatial_mean=None,
        spatial_cov=None,
        spectral_mean=None,
        spectral_cov=None,
        spectral_guess=None,
        spectral_lambda=None,
        spectral_maxiter=None,
        spectral_eps=None,
        spectral_tol=None,
        spectral_method=None,
        normalized=True,
        baseline=None,
        baseline_var=None,
        fix_spectrum=False,
        fix_map=False,
        logT0=None,
        logTf=None,
        nlogT=None,
        quiet=False,
    ):
        # --------------------------
        # ---- Process defaults ----
        # --------------------------

        if flux_err is None:
            flux_err = 1e-4
        if spatial_mean is None:
            spatial_mean = np.zeros(self.Ny)
            spatial_mean[0] = 1.0
        if spatial_cov is None:
            spatial_cov = 1e-4
        if baseline_var is None:
            baseline_var = 1e-2
        if logT0 is None:
            logT0 = 2
        if logTf is None:
            logTf = 0
        if nlogT is None:
            nlogT = 50
        else:
            nlogT = max(1, nlogT)
        if spectral_mean is None:
            spectral_mean = 1.0
        if spectral_cov is None:
            spectral_cov = 1e-3
        if spectral_lambda is None:
            spectral_lambda = 1e5
        if spectral_maxiter is None:
            spectral_maxiter = 100
        if spectral_method is None:
            spectral_method = "L2"
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

        # Spatial cov may be a scalar, a vector, a matrix (Ny, Ny),
        # or a list of those. Invert it and reshape to a matrix of
        # shape (Ny, nc) (inverse variances) or a tensor of shape
        # (Ny, Ny, nc) (nc separate inverse covariance matrices)
        if type(spatial_cov) not in (list, tuple):
            # Use the same covariance for all components
            spatial_cov = [spatial_cov for n in range(self.nc)]
        else:
            # Check that we have one covariance per component
            assert len(spatial_cov) == self.nc
        spatial_inv_cov = [None for n in range(self.nc)]
        ndim = np.array(spatial_cov[0]).ndim
        for n in range(self.nc):
            spatial_cov[n] = np.array(spatial_cov[n])
            assert spatial_cov[n].ndim == ndim
            if spatial_cov[n].ndim < 2:
                spatial_inv_cov[n] = np.reshape(
                    np.ones(self.Ny) / spatial_cov[n], (-1, 1)
                )
                spatial_cov[n] = np.reshape(
                    np.ones(self.Ny) * spatial_cov[n], (-1, 1)
                )
            else:
                cho = cho_factor(spatial_cov[n])
                inv = cho_solve(cho, np.eye(self.Ny))
                spatial_inv_cov[n] = np.reshape(inv, (self.Ny, self.Ny, 1))
                spatial_cov[n] = np.reshape(
                    spatial_cov[n], (self.Ny, self.Ny, 1)
                )

        # Tensor of nc (inverse) variance vectors or covariance matrices
        self.spatial_cov = np.concatenate(spatial_cov, axis=-1)
        self.spatial_inv_cov = np.concatenate(spatial_inv_cov, axis=-1)

        # Baseline must be a vector (nt,)
        if baseline is not None:
            assert np.array_equal(
                np.shape(baseline), np.array([self.nt])
            ), "Invalid shape for `baseline`."
            self.baseline = baseline
        else:
            self.baseline = None

        # Spectral mean must be a scalar, a vector (nw0), or a list of those
        # Interpolate it to the internal grid (nw0_) and reshape to (nc, nw0_)
        if type(spectral_mean) not in (list, tuple):
            # Use the same mean for all components
            spectral_mean = [spectral_mean for n in range(self.nc)]
        else:
            # Check that we have one mean per component
            assert len(spectral_mean) == self.nc
        for n in range(self.nc):
            spectral_mean[n] = np.array(spectral_mean[n])
            assert spectral_mean[n].ndim < 2
            spectral_mean[n] = np.reshape(
                spectral_mean[n] * np.ones(self.nw0), (-1, 1)
            )
            spectral_mean[n] = self.S0e2i.dot(spectral_mean[n]).T
        self.spectral_mean = np.concatenate(spectral_mean, axis=0)

        # Spectral cov may be a scalar, a vector, a matrix (nw0, nw0),
        # or a list of those. Interpolate it to the internal grid,
        # then invert it and reshape to a matrix of
        # shape (nc, nw0_) (inverse variances) or a tensor of shape
        # (nc, nw0_, nw0_) (nc separate inverse covariance matrices)
        if type(spectral_cov) not in (list, tuple):
            # Use the same covariance for all components
            spectral_cov = [spectral_cov for n in range(self.nc)]
        else:
            # Check that we have one covariance per component
            assert len(spectral_cov) == self.nc
        spectral_inv_cov = [None for n in range(self.nc)]
        ndim = np.array(spectral_cov[0]).ndim
        for n in range(self.nc):
            spectral_cov[n] = np.array(spectral_cov[n])
            assert spectral_cov[n].ndim == ndim
            if spectral_cov[n].ndim < 2:
                if spectral_cov[n].ndim == 0:
                    cov = np.ones(self.nw0_) * spectral_cov[n]
                else:
                    cov = self.S0e2i.dot(spectral_cov[n])
                inv = 1.0 / cov
                spectral_inv_cov[n] = np.reshape(inv, (1, -1))
                spectral_cov[n] = np.reshape(cov, (1, -1))
            else:
                cov = self.S0e2i.dot(self.S0e2i.dot(spectral_cov[n]).T).T
                cov[np.diag_indices_from(cov)] += spectral_eps
                cho = cho_factor(cov)
                inv = cho_solve(cho, np.eye(self.nw0_))
                spectral_inv_cov[n] = np.reshape(
                    inv, (1, self.nw0_, self.nw0_)
                )
                spectral_cov[n] = np.reshape(cov, (1, self.nw0_, self.nw0_))

        # Tensor of nc (inverse) variance vectors or covariance matrices
        self.spectral_cov = np.concatenate(spectral_cov, axis=0)
        self.spectral_inv_cov = np.concatenate(spectral_inv_cov, axis=0)

        # Spectral guess must be a scalar, a vector (nw0), or a list of those
        # Interpolate it to the internal grid (nw0_) and reshape to (nc, nw0_)
        if spectral_guess is not None:
            if type(spectral_guess) not in (list, tuple):
                # Use the same guess for all components
                spectral_guess = [spectral_guess for n in range(self.nc)]
            else:
                # Check that we have one mean per component
                assert len(spectral_guess) == self.nc
            for n in range(self.nc):
                spectral_guess[n] = np.array(spectral_guess[n])
                assert spectral_guess[n].ndim < 2
                spectral_guess[n] = np.reshape(
                    spectral_guess[n] * np.ones(self.nw0), (-1, 1)
                )
                spectral_guess[n] = self.S0e2i.dot(spectral_guess[n]).T
            self.spectral_guess = np.concatenate(spectral_guess, axis=0)
        else:
            self.spectral_guess = None

        # Tempering schedule
        if nlogT == 1:
            self.T = np.array([10 ** logTf])
        elif logT0 == logTf:
            self.T = logTf * np.ones(nlogT)
        else:
            self.T = np.logspace(logT0, logTf, nlogT)

        # Ingest the remaining params
        self.spectral_lambda = spectral_lambda  # must be scalar
        self.spectral_maxiter = spectral_maxiter
        self.spectral_eps = spectral_eps
        self.spectral_tol = spectral_tol
        self.spectral_method = spectral_method
        self.normalized = normalized
        self.baseline_var = baseline_var
        self.fix_spectrum = fix_spectrum
        self.fix_map = fix_map
        self.quiet = quiet

        # Are we lucky enough to do a purely linear solve for the map?
        self.linear = (not self.normalized) or (self.baseline is not None)

    def solve_for_map_linear(self, T=1, baseline_var=0):
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

        # Baseline correction
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

        else:

            # Easy!
            flux = self.flux

        # Factorized data covariance
        if baseline_var == 0:
            cho_C = np.reshape(np.sqrt(T) * flux_err, (-1,))
        else:
            cho_C = block_diag(
                *[
                    cho_factor(
                        T * np.diag(flux_err.reshape(self.nt, self.nw)[n] ** 2)
                        + baseline_var
                    )
                    for n in range(self.nt)
                ]
            )

        # Unroll the data into a vector
        flux = np.reshape(flux, (-1,))

        # Solve the L2 problem
        mean, cho_ycov = greedy_linalg.solve(self.S, flux, cho_C, mu, invL)
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
                cho_C = block_diag(
                    *[
                        cho_factor(
                            self.T[i]
                            * self.flux_err ** 2
                            * np.eye(self.nw)
                            * baseline.reshape(self.nt, self.nw)[n] ** 2
                            + self.baseline_var
                        )
                        for n in range(self.nt)
                    ]
                )
            elif self.flux_err.ndim == 2:
                cho_C = block_diag(
                    *[
                        cho_factor(
                            self.T[i]
                            * np.diag(self.flux_err[n] ** 2)
                            * baseline.reshape(self.nt, self.nw)[n] ** 2
                            + self.baseline_var
                        )
                        for n in range(self.nt)
                    ]
                )
            else:
                raise ValueError("Invalid shape for `flux_err`.")

            # Solve the L2 problem for the Ylm coeffs
            y_flat, cho_ycov = greedy_linalg.solve(
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

    def solve_for_spectrum_linear(self):
        """
        Solve for `spectrum_` conditioned on the current map.

        """
        # Unroll the data into a vector
        if self.normalized:

            # Un-normalize the data with the current baseline
            if self.baseline is None:
                baseline = np.dot(self.C, self.y.T.reshape(-1))
                baseline = np.reshape(baseline, (self.nt, 1))
            else:
                baseline = np.reshape(self.baseline, (self.nt, 1))
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
        mu = self.spectral_mean.reshape(-1)
        if self.spectral_inv_cov.ndim == 2:
            invL = self.spectral_inv_cov.reshape(-1)
            invLmu = invL * mu
        else:
            invL = block_diag(*[inv for inv in self.spectral_inv_cov])
            invLmu = np.dot(invL, mu)

        # Compute M^T C^-1 M
        # TODO: Figure out how to do this with a single convolution
        KInv = self.dotMT(np.transpose(self.dotMT(CInv)))

        if self.spectral_method.upper() == "L2":

            # Compute M^T C^-1 f
            term = self.dotMT(np.dot(CInv, flux)).reshape(-1)

            # Solve the L2 problem
            if invL.ndim == 1:
                KInv[np.diag_indices_from(KInv)] += invL
            else:
                KInv += invL
            choK = cho_factor(KInv)
            spectrum_ = cho_solve(choK, term + invLmu)

        elif self.spectral_method.upper() == "L1":

            # Compute M^T C^-1 f
            mean_flux = self.dotM(mu).reshape(-1)
            term = self.dotMT(np.dot(CInv, flux - mean_flux)).reshape(-1)

            # Solve the L1 problem
            spectrum_ = mu + self.L1(
                KInv,
                term,
                np.array(self.spectral_lambda),
                self.spectral_maxiter,
                np.array(self.spectral_eps),
                np.array(self.spectral_tol),
            )
            choK = None

        else:

            raise ValueError("Invalid `spectral_method`.")

        # Store
        self.spectrum_ = np.reshape(spectrum_, (self.nc, self.nw0_))
        self.cho_scov = cho_factor(cho_solve(choK, np.eye(choK.shape[0])))

    def solve_for_everything_bilinear(self):
        """
        Solve for both the map and the spectrum.

        """
        if self.spectral_guess is None:
            # First, deconvolve the flux to obtain a guess for the spectrum
            # We subtract the convolved mean spectrum and add the deconvolved
            # residuals to each of the component spectral means to obtain
            # a rough guess for all components. This is by no means optimal.
            f = self.Se2i.dot(np.mean(self.flux, axis=0))
            f /= f[self.continuum_idx]
            mu = self.spectral_mean.T
            f -= np.mean(np.dot(self.KT0, mu), axis=1)
            CInv = np.dot(self.KT0.T, self.KT0) / np.mean(self.flux_err) ** 2
            term = np.dot(self.KT0.T, f) / np.mean(self.flux_err) ** 2
            self.spectrum_ = mu.T + self.L1(
                CInv,
                term,
                np.array(self.spectral_lambda),
                self.spectral_maxiter,
                np.array(self.spectral_eps),
                np.array(self.spectral_tol),
            )
        else:
            self.spectrum_ = self.spectral_guess
        self.meta["spectrum_guess"] = self.spectrum_

        # Assume a unit baseline guess
        self.baseline = np.ones(self.nt)

        # Iterate
        for i in tqdm(range(len(self.T)), disable=self.quiet):

            # Solve for the map
            self._S = None
            if self.linear:

                self.solve_for_map_linear(T=self.T[i])

            else:

                self.solve_for_map_linear(
                    T=self.T[i], baseline_var=self.baseline_var
                )
                self.baseline = np.dot(self.C, self.y.T.reshape(-1))

            # Solve for the spectrum
            self.solve_for_spectrum_linear()

    def solve_bilinear(self, flux, theta, y, spectrum_, veq, inc, u, **kwargs):
        """
        Solve the linear problem for the spatial and/or spectral map
        given a spectral timeseries.

        """
        # Start fresh
        self.reset()

        # Parse the inputs
        self.process_inputs(flux, **kwargs)
        self.theta = theta
        self.veq = veq
        self.inc = inc
        self.y = y
        self.u = u
        self.cho_ycov = None
        self.spectrum_ = spectrum_
        self.cho_scov = None

        if self.fix_spectrum:

            # Solve for the map conditioned on the spectrum
            if self.linear:

                # The problem is exactly linear!
                self.solve_for_map_linear()

            else:

                # The problem is linear conditioned on a baseline
                # guess, which we refine iteratively
                self.solve_for_map_tempered()

        elif self.fix_map:

            # Solve for the spectrum conditioned on the map
            self.solve_for_spectrum_linear()

        else:

            # Solve for both the map and the spectrum
            self.solve_for_everything_bilinear()

        # Update the metadata with the results and return everything
        self.meta["y"] = self.y
        self.meta["cho_ycov"] = self.cho_ycov
        self.meta["spectrum_"] = self.spectrum_
        self.meta["cho_scov"] = self.cho_scov
        return self.meta

    def solve_nonlinear(
        self,
        flux,
        theta,
        y,
        spectrum_,
        veq,
        inc,
        u,
        lr=2e-5,
        niter=1000,
        **kwargs
    ):
        """
        Solve the nonlinear problem for the spatial and/or spectral map
        given a spectral timeseries.

        NOTE: Experimental, not well tested.

        """
        # Start fresh
        self.reset()

        # Parse the inputs
        self.process_inputs(flux, **kwargs)

        # The variables we're optimizing
        tt_y = theano.shared(y)
        tt_spectrum_ = theano.shared(spectrum_)
        tt_vars = []
        if not self.fix_map:
            tt_vars += [tt_y]
        if not self.fix_spectrum:
            tt_vars += [tt_spectrum_]

        # Compute the exact model
        tt_model = self.get_flux_from_dotconv(
            inc, theta, veq, u, tt_y, tt_spectrum_
        )

        # Interpolate to the output grid
        if self.interp:
            tt_model = ts.dot(tt_model, self.Si2eTr)

        # Remove the baseline?
        if self.normalized:
            if self.baseline is None:
                tt_model /= tt.reshape(
                    tt_model[:, self.continuum_idx], (self.nt, 1)
                )
            else:
                tt_model /= tt.reshape(self.baseline, (self.nt, 1))

        # The likelihood term
        tt_loss = tt.sum((self.flux - tt_model) ** 2 / self.flux_err ** 2)

        # The spatial prior
        tt_x = tt.reshape(tt.transpose(tt_y), (-1, 1))
        mu = self.spatial_mean.T.reshape(-1, 1)
        tt_r = tt_x - mu
        if self.spatial_cov.ndim == 2:
            invL = np.reshape(np.transpose(self.spatial_inv_cov), (-1, 1))
            tt_loss += tt.dot(tt.transpose(tt_r), tt_r * invL)[0, 0]
        else:
            L = block_diag(
                *[self.spatial_cov[:, :, n] for n in range(self.nc)]
            )
            tt_loss += tt.dot(
                tt.transpose(tt_r), tt_cho_solve(tt_cho_factor(L), tt_r)
            )[0, 0]

        # The spectral prior
        tt_x = tt.reshape(tt_spectrum_, (-1, 1))
        mu = self.spectral_mean.reshape(-1, 1)
        tt_r = tt_x - mu
        if self.spectral_cov.ndim == 2:
            invL = np.reshape(self.spectral_inv_cov, (-1, 1))
            tt_loss += tt.dot(tt.transpose(tt_r), tt_r * invL)[0, 0]
        else:
            L = block_diag(*[cov for cov in self.spectral_cov])
            tt_loss += tt.dot(
                tt.transpose(tt_r), tt_cho_solve(tt_cho_factor(L), tt_r)
            )[0, 0]

        # Compile the NAdam optimizer and iterate
        tt_updates = nadam(tt_loss, tt_vars, lr=lr)
        train = theano.function(
            [], [tt_y, tt_spectrum_, tt_loss], updates=tt_updates
        )

        # Iterate
        best_loss = np.inf
        best_y = np.array(y)
        best_spectrum_ = np.array(spectrum_)
        loss = np.zeros(niter)
        for n in tqdm(range(niter), disable=self.quiet):
            curr_y, curr_spectrum_, loss[n] = train()
            if loss[n] < best_loss:
                best_loss = loss[n]
                best_y = curr_y
                best_spectrum_ = curr_spectrum_

        # Record the best values and return
        self.meta["y"] = best_y
        self.meta["spectrum_"] = best_spectrum_
        self.meta["loss"] = loss

        return self.meta
