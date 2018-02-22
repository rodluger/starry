"""Plot the Ylms on the surface of the sphere."""
from starry import starry
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np


class animated():
    """Plot an animated GIF showing rotation of the Ylms."""

    def __init__(self, lmax=5, res=300, dpi=100, fps=10, frames=50,
                 u=[0., 1., 0.]):
        """Initialize."""
        self.lmax = lmax
        self.s = starry(lmax)
        self.res = res
        self.u = np.array(u)
        self.frames = frames

        # Set up the plot
        self.fig, self.ax = pl.subplots(self.lmax + 1, 2 * self.lmax + 1,
                                        figsize=(9, 6))
        self.fig.subplots_adjust(hspace=0)
        for axis in self.ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
        for l in range(self.lmax + 1):
            self.ax[l, 0].set_ylabel(r"$l = %d$" % l,
                                     rotation='horizontal',
                                     labelpad=30, y=0.38,
                                     fontsize=12)
        for j, m in enumerate(range(-self.lmax, self.lmax + 1)):
            self.ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=30,
                                      fontsize=12)

        self.flux = np.empty((self.res, self.res), dtype=float)

        # Loop over the orders and degrees
        self.img = []
        for i, l in enumerate(range(self.lmax + 1)):
            for j, m in enumerate(range(-l, l + 1)):

                # Offset the index for centered plotting
                j += self.lmax - l

                # Compute the spherical harmonic
                self.s[:] = 0
                self.s[l, m] = 1
                self.flux = self.s.render(self.u, 0, res=self.res)

                # Plot the spherical harmonic
                img = self.ax[i, j].imshow(self.flux, cmap='plasma',
                                           interpolation="none",
                                           origin="lower")
                self.img.append(img)

        # Set up the animation
        self.theta = np.linspace(0, 2 * np.pi, frames, endpoint=False)
        self.animation = animation.FuncAnimation(self.fig, self.animate,
                                                 frames=self.frames,
                                                 interval=50,
                                                 repeat=True, blit=True)

        # Save
        self.animation.save('ylms.gif', writer='imagemagick',
                            fps=fps, dpi=dpi)
        pl.close()

    def animate(self, j):
        """Run the animation."""
        print("Rendering frame %d/%d..." % (j + 1, self.frames))
        # Rotate the spherical harmonics
        n = 0
        theta = self.theta[j]
        for i, l in enumerate(range(self.lmax + 1)):
            for j, m in enumerate(range(-l, l + 1)):
                self.s[:] = 0
                self.s[l, m] = 1
                self.flux = self.s.render(self.u, theta, res=self.res)
                self.img[n].set_data(self.flux)
                n += 1
        return self.img


def static(lmax=5, res=300):
    """Plot a static PDF figure."""
    # Set up the plot
    fig, ax = pl.subplots(lmax + 1, 2 * lmax + 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
    for l in range(lmax + 1):
        ax[l, 0].set_ylabel(r"$l = %d$" % l,
                            rotation='horizontal',
                            labelpad=30, y=0.38,
                            fontsize=12)
    for j, m in enumerate(range(-lmax, lmax + 1)):
        if m < 0:
            ax[-1, j].set_xlabel(r"$m {=} \mathrm{-}%d$" % -m,
                                 labelpad=30, fontsize=11)
        else:
            ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=30, fontsize=11)

    # Plot it
    flux = np.empty((res, res), dtype=float)
    s = starry(lmax)

    # Loop over the orders and degrees
    for i, l in enumerate(range(lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):

            # Offset the index for centered plotting
            j += lmax - l

            # Compute the spherical harmonic
            # with no rotation
            u = np.array([1., 0., 0.])
            theta = 0.
            s[:] = 0
            s[l, m] = 1
            flux = s.render(u, theta, res=res)

            # Plot the spherical harmonic
            ax[i, j].imshow(flux, cmap='plasma',
                            interpolation="none", origin="lower",
                            extent=(-1, 1, -1, 1))
            ax[i, j].set_xlim(-1.1, 1.1)
            ax[i, j].set_ylim(-1.1, 1.1)

    # Save!
    fig.savefig("ylms.pdf", bbox_inches="tight")
    pl.close()


if __name__ == "__main__":
    static()
