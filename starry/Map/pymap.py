# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..extensions import RAxisAngle
from .sht import image2map, healpix2map, array2map
from IPython.display import HTML


__all__ = ["PythonMapBase"]


class PythonMapBase(object):
    """

    """

    def render(self, theta=0, res=300, projection="ortho", **kwargs):
        """

        """
        # Type-specific kwargs
        if projection.lower().startswith("rect"):
            projection = "rect"
            nframes = 1
            model_kwargs = dict()
        elif projection.lower().startswith("ortho"):
            projection = "ortho"
            if hasattr(theta, "__len__"):
                nframes = len(theta)
            else:
                nframes = 1
            model_kwargs = dict(theta=theta)
        else:
            raise ValueError("Invalid projection. Allowed projections are " +
                             "`rectangular` and `orthographic` (default).")

        # Are we modeling time variability?
        if self._temporal:
            t = kwargs.pop("t", 0.0)
            if hasattr(t, "__len__"):
                nframes = max(nframes, len(t))
            model_kwargs["t"] = t

        # Are we modeling reflected light?
        if self._reflected:
            source = kwargs.pop("source", [[-1.0, 0.0, 0.0] for n in range(nframes)])
            if source is None:
                # If explicitly set to `None`, re-run this
                # function on an *emitted* light map!
                from .. import Map
                if self._temporal:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg,
                              multi=self.multi, nt=self.nt)
                    map[:, :, :] = self[:, :, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection, t=t)
                elif self._spectral:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg,
                              multi=self.multi, nw=self.nw)
                    map[:, :, :] = self[:, :, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection)
                else:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg,
                              multi=self.multi)
                    map[:, :] = self[:, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection)
            else:
                source = np.ascontiguousarray(source)
                if len(source.shape) == 2:
                    nframes = max(nframes, len(source))
                model_kwargs["source"] = source

        # Are we doing wavelength dependence?
        if self._spectral:
            assert nframes == 1, "Spectral map rotation cannot be animated."

        if projection == "rect":

            # Disable limb darkening
            if self.udeg:
                u_copy = np.array(self[1:])
                self[1:] = 0

            # Generate the lat/lon grid for one hemisphere
            lon = np.linspace(-np.pi, np.pi, res)
            lat = np.linspace(1e-3, np.pi / 2, res // 2)
            lon, lat = np.meshgrid(lon, lat)
            x = np.sin(np.pi / 2 - lat) * np.cos(lon - np.pi / 2)
            y = np.sin(np.pi / 2 - lat) * np.sin(lon - np.pi / 2)

            # Rotate so we're looking down the north pole
            map_axis = np.array(self.axis)
            alpha = np.arccos(np.dot(map_axis, [0, 0, 1])) * 180 / np.pi
            u = np.cross(map_axis, [0, 0, 1])
            self.axis = u
            self.rotate(alpha)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle(u, alpha)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source

            # Compute the northern hemisphere map
            self.axis = [0, 0, 1]
            Z_north = np.array(self.intensity(x=x, y=y, **model_kwargs))
            if self._spectral:
                Z_north = Z_north.reshape(res // 2, res, self.nw)
                Z_north = np.moveaxis(Z_north, -1, 0)
            else:
                Z_north = Z_north.reshape(nframes, res // 2, res)

            # Flip the planet around
            self.axis = [1, 0, 0]
            self.rotate(180)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle([1, 0, 0], 180)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source
            
            # Compute the southern hemisphere map
            self.axis = [0, 0, -1]
            Z_south = np.array(self.intensity(x=-x, y=-y, **model_kwargs))
            if self._spectral:
                Z_south = Z_south.reshape(res // 2, res, self.nw)
                Z_south = np.moveaxis(Z_south, -1, 0)
            else:
                Z_south = Z_south.reshape(nframes, res // 2, res)
            Z_south = np.flip(Z_south, axis=(1, 2))

            # Join them
            Z = np.concatenate((Z_south, Z_north), axis=1)

            # Undo all the rotations
            self.axis = [1, 0, 0]
            self.rotate(-180)
            self.axis = u
            self.rotate(-alpha)
            self.axis = map_axis

            # Re-enable limb darkening
            if self.udeg:
                self[1:] = u_copy

        else:

            # Create a grid of X and Y and construct the linear model
            x, y = np.meshgrid(np.linspace(-1, 1, res), 
                               np.linspace(-1, 1, res))
            Z = np.array(self.intensity(x=x, y=y, **model_kwargs))
            if self._spectral:
                Z = Z.reshape(res, res, self.nw)
                Z = np.moveaxis(Z, -1, 0)
            else:
                Z = Z.reshape(nframes, res, res)

        return np.squeeze(Z)

    def show(self, Z=None, cmap="plasma", projection="ortho", 
             grid=True, **kwargs):
        """

        """
        # Render the map
        if Z is None:
            Z = self.render(projection=projection, **kwargs)
        if len(Z.shape) == 3:
            nframes = Z.shape[0]
        else:
            nframes = 1
            Z = [Z]

        # Are we doing wavelength dependence?
        if self._spectral:
            animated = True
        else:
            animated = (nframes > 1)

        # Latitude grid lines
        latlines = [-60, -30, 0, 30, 60]
        lonlines = np.linspace(-180, 180, 13)

        if projection == "rect":
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(7, 3.75))
            extent = (-180, 180, -90, 90)

            if grid:
                for lat in latlines:
                    ax.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
                for lon in lonlines:
                    ax.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
            ax.set_xticks(lonlines)
            ax.set_yticks(latlines)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")

        else:
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(3, 3))
            ax.axis('off')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            extent = (-1, 1, -1, 1)

            # Plot the lat/lon grid lines
            if grid:
                
                # Body outline
                x = np.linspace(-1, 1, 10000)
                y = np.sqrt(1 - x ** 2)
                ax.plot(x, y, 'k-', alpha=1, lw=1)
                ax.plot(x, -y, 'k-', alpha=1, lw=1)

                # Angular quantities
                ci = np.cos(self.inc * np.pi / 180)
                si = np.sin(self.inc * np.pi / 180)
                co = np.cos(self.obl * np.pi / 180)
                so = np.sin(self.obl * np.pi / 180)

                # Mark the pole
                if self.inc < 90:
                    x = si * so
                    y = si * co
                elif self.inc > 90:
                    x = -si * so
                    y = -si * co
                ax.plot(x, y, 'ko', ms=2, alpha=0.5)

                # Latitude lines
                for lat in latlines:

                    # Figure out the equation of the ellipse
                    y0 = np.sin(lat * np.pi / 180) * si
                    a = np.cos(lat * np.pi / 180)
                    b = a * ci
                    x = np.linspace(-a, a, 10000)
                    y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
                    y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

                    # Mask lines on the backside
                    if (si != 0):
                        if self.inc > 90:
                            ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                            y1[y1 < ymax] = np.nan
                            ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                            y2[y2 < ymax] = np.nan
                        else:
                            ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                            y1[y1 > ymax] = np.nan
                            ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                            y2[y2 > ymax] = np.nan

                    # Rotate them
                    for y in (y1, y2):
                        xr = -x * co + y * so
                        yr = x * so + y * co
                        ax.plot(xr, yr, 'k-', lw=0.5, alpha=0.5, zorder=100)

                # Longitude lines
                for lon in lonlines:
                    # Viewed at i = 90
                    b = np.sin(lon * np.pi / 180)
                    y = np.linspace(-1, 1, 1000)
                    x = b * np.sqrt(1 - y ** 2)
                    z = np.sqrt(np.abs(1 - x ** 2 - y ** 2))

                    # Rotate by the inclination
                    R = RAxisAngle([1, 0, 0], 90 - self.inc)
                    v = np.vstack((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
                    x, y1, _ = np.dot(R, v)
                    v[2] *= -1
                    _, y2, _ = np.dot(R, v)

                    # Mask lines on the backside
                    if (si != 0):
                        if self.inc < 90:
                            imax = np.argmax(x ** 2 + y1 ** 2)
                            y1[:imax + 1] = np.nan
                            imax = np.argmax(x ** 2 + y2 ** 2)
                            y2[:imax + 1] = np.nan
                        else:
                            imax = np.argmax(x ** 2 + y1 ** 2)
                            y1[imax:] = np.nan
                            imax = np.argmax(x ** 2 + y2 ** 2)
                            y2[imax:] = np.nan

                    # Rotate them
                    for y in (y1, y2):
                        xr = -x * co + y * so
                        yr = x * so + y * co
                        ax.plot(xr, yr, 'k-', lw=0.5, alpha=0.5, zorder=100)

        # Plot the first frame of the image
        img = ax.imshow(Z[0], origin="lower", 
                        extent=extent, cmap=cmap,
                        interpolation="none",
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z), 
                        animated=animated)
        if projection == "rect":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, ax=ax, cax=cax)

        # Display or save the image / animation
        if animated:
            interval = kwargs.pop("interval", 75)
            mp4 = kwargs.pop("mp4", None)
            
            def updatefig(i):
                img.set_array(Z[i])
                return img,

            ani = FuncAnimation(fig, updatefig, interval=interval,
                                blit=False, frames=len(Z))

            # Business as usual
            if (mp4 is not None) and (mp4 != ""):
                if mp4.endswith(".mp4"):
                    mp4 = mp4[:-4]
                ani.save('%s.mp4' % mp4, writer='ffmpeg')
                plt.close()
            else:
                try:
                    if 'zmqshell' in str(type(get_ipython())):
                        plt.close()
                        display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()
        else:
            plt.show()

    def flux(self, *args, **kwargs):
        """

        """
        # This is already implemented for limb-darkened maps
        if (self._limbdarkened):
            return super(PythonMapBase, self).flux(*args, **kwargs)

        if kwargs.get("gradient", False):
            # Get the design matrix and its gradient
            X, grad = self.linear_flux_model(*args, **kwargs)
            
            # The dot product with `y` gives us the flux
            f = np.dot(X, self.y)
            for key in grad.keys():
                grad[key] = np.dot(grad[key], self.y)

            # Add in the gradient with respect to `y`, but
            # first remove inds where `l = m = 0`
            lgtr0 = np.ones(self.Ny * self.nt, dtype=bool)
            for i in range(self.nt):
                lgtr0[i * self.Ny] = False
            grad['y'] = X[:, lgtr0].T

            # Copy df/dy to each wavelength bin
            if self._spectral:
                grad['y'] = np.tile(grad['y'][:, :, np.newaxis], 
                                    (1, 1, self.nw))

            return f, grad
        else:
            # The flux is just the dot product with the design matrix
            return np.dot(self.linear_flux_model(*args, **kwargs), self.y)

    def __call__(self, *args, **kwargs):
        """

        """
        return self.intensity(*args, **kwargs)
    
    def load(self, image, ydeg=None, healpix=False, col=0, **kwargs):
        """Load an image, array, or healpix map."""
        if self._limbdarkened:
            raise NotImplementedError("The `load` method is not " + 
                                      "implemented for limb-darkened maps.")

        # Check the degree
        if ydeg is None:
            ydeg = self.ydeg
        assert (ydeg <= self.ydeg) and (ydeg > 0), \
            "Invalid spherical harmonic degree."
        
        # Is this a file name?
        if type(image) is str:
            y = image2map(image, lmax=ydeg, **kwargs)
        # or is it an array?
        elif (type(image) is np.ndarray):
            if healpix:
                y = healpix2map(image, lmax=ydeg, **kwargs)
            else:
                y = array2map(image, lmax=ydeg, **kwargs)
        else:
            raise ValueError("Invalid `image` value.")
        
        # Ingest the coefficients
        if self._spectral or self._temporal:
            self[1:, :, :] = 0
            self[:ydeg + 1, :, col] = y
        else:
            self[1:, :] = 0
            self[:ydeg + 1, :] = y
    
    def project(self):
        """
        TODO: NOT A PERMANENT SOLUTION...

        """
        axis = np.array(self.axis)
        theta = np.arccos(np.dot([0, 1, 0], axis)) * 180 / np.pi
        self.axis = np.cross([0, 1, 0], axis)
        self.rotate(theta)
        self.axis = axis
    
    def deproject(self):
        """
        TODO: NOT A PERMANENT SOLUTION...

        """
        axis = np.array(self.axis)
        theta = np.arccos(np.dot([0, 1, 0], axis)) * 180 / np.pi
        self.axis = np.cross([0, 1, 0], axis)
        self.rotate(-theta)
        self.axis = axis