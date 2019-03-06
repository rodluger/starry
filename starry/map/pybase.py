# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..extensions import RAxisAngle
from .mapsum import MapSum

__all__ = ["PythonMapBase"]


class PythonMapBase(object):
    """

    """

    def render(self, theta=0, res=300, projection="ortho", 
               rotate_if_rect=False, **kwargs):
        """

        """
        # Type-specific kwargs
        if projection.lower().startswith("rect"):
            projection = "rect"
            if rotate_if_rect and hasattr(theta, "__len__"):
                nframes = len(theta)
                model_kwargs = dict(theta=theta)
            else:
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

        # Adjust `res` so that it's divisible by 4
        res += (4 - (res % 4))

        # Are we modeling time variability?
        if self._temporal:
            t = kwargs.pop("t", 0.0)
            if hasattr(t, "__len__"):
                nframes = max(nframes, len(t))
            model_kwargs["t"] = t

        # Are we modeling reflected light?
        if self._reflected:
            source = np.array(kwargs.pop("source", [-1.0, 0.0, 0.0]))
            if len(source.shape) == 2:
                nframes = max(nframes, len(source))
            model_kwargs["source"] = source

        # Are we doing wavelength dependence?
        if self._spectral:
            assert nframes == 1, "Spectral map rotation cannot be animated."

        if projection == "rect":

            # Generate the lat/lon grid for one hemisphere
            lon = np.linspace(-np.pi, np.pi, res)
            lat = np.linspace(1e-3, np.pi / 2, res // 2)
            lon, lat = np.meshgrid(lon, lat)
            x = np.sin(np.pi / 2 - lat) * np.cos(lon - np.pi / 2)
            y = np.sin(np.pi / 2 - lat) * np.sin(lon - np.pi / 2)

            # Rotate so we're looking down the north pole
            map_axis = np.array(self.axis)
            sinalpha = np.sqrt(self.axis[0] ** 2 + self.axis[1] ** 2)
            cosalpha = self.axis[2]
            u = np.array([self.axis[1], self.axis[0], 0]) / sinalpha
            alpha = (180 / np.pi) * np.arctan2(sinalpha, cosalpha)
            self.axis = u
            self.rotate(alpha)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle(u, alpha)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source

            # Compute the linear model
            self.axis = [0, 0, 1]
            X = self.linear_intensity_model(x=x, y=y, **model_kwargs)

            # Compute the northern hemisphere map
            if self._spectral:
                Z_north = np.array([np.dot(X, self.y[:, n]).reshape(res // 2, res) 
                                    for n in range(self.nw)])
            else:
                Z_north = np.dot(X, self.y).reshape(nframes, res // 2, res)

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
            
            # Compute the linear model
            self.axis = [0, 1e-10, -1] # TODO: Bug when axis = [0, 0, -1]
            X = self.linear_intensity_model(x=-x, y=-y, **model_kwargs)

            # Compute the southern hemisphere map
            if self._spectral:
                Z_south = np.array([np.dot(X, self.y[:, n]).reshape(res // 2, res) 
                                    for n in range(self.nw)])
            else:
                Z_south = np.dot(X, self.y).reshape(nframes, res // 2, res)
            Z_south = np.flip(Z_south, axis=(1, 2))

            # Join them
            Z = np.concatenate((Z_south, Z_north), axis=1)

            # Undo all the rotations
            self.axis = [1, 0, 0]
            self.rotate(-180)
            self.axis = u
            self.rotate(-alpha)
            self.axis = map_axis

        else:

            # Create a grid of X and Y and construct the linear model
            x, y = np.meshgrid(np.linspace(-1, 1, res), 
                               np.linspace(-1, 1, res))
            X = self.linear_intensity_model(x=x, y=y, **model_kwargs)
            if self._spectral:
                Z = np.moveaxis(
                        np.dot(X, self.y).reshape(res, res, self.nw), 
                        -1, 0)
            else:
                Z = np.dot(X, self.y).reshape(nframes, res, res)

        return Z

    def show(self, Z=None, cmap="plasma", projection="ortho", **kwargs):
        """

        """
        # Render the map
        if Z is None:
            Z = self.render(projection=projection, **kwargs)
        if len(Z.shape) == 3:
            nframes = Z.shape[0]
        else:
            nframes = 1

        # Are we doing wavelength dependence?
        if self._spectral:
            animated = True
        else:
            animated = (nframes > 1)

        if projection == "rect":
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(6, 3))
            extent = (-180, 180, -90, 90)
        else:
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(3, 3))
            ax.axis('off')
            extent = (-1, 1, -1, 1)

        # Plot the first frame of the image
        img = ax.imshow(Z[0], origin="lower", 
                        extent=extent, cmap=cmap,
                        interpolation="none",
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z), 
                        animated=animated)
        
        # Display or save the image / animation
        if animated:
            interval = kwargs.pop("interval", 75)
            gif = kwargs.pop("gif", None)
            
            def updatefig(i):
                img.set_array(Z[i])
                return img,

            ani = FuncAnimation(fig, updatefig, interval=interval,
                                blit=False, frames=len(Z))

            # TODO: Jupyter override

            # Business as usual
            if (gif is not None) and (gif != ""):
                if gif.endswith(".gif"):
                    gif = gif[:-4]
                ani.save('%s.gif' % gif, writer='imagemagick')
            else:
                plt.show()
            plt.close()
        else:
            plt.show()
    
    def __add__(self, other):
        """

        """
        return MapSum(self) + other
    

    def flux(self, **kwargs):
        """

        """
        return np.dot(self.linear_flux_model(**kwargs), self.y)