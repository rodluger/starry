"""Test the STARRY rotation routines."""
from starry import R
from starry.basis import evaluate_poly, y2p
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation

# Construct the vector for Y_{1,-1}
y = [0, 1, 0, 0]


class animated():
    """Plot an animated GIF showing the rotation of a Ylm."""

    def __init__(self, y=[0, 1, 0, 0], res=300, dpi=100, fps=10, frames=50):
        """Initialize."""
        self.y = y
        self.res = res
        self.frames = frames
        self.u = np.array([[[1, 0, 0],
                            [0, 1, 0]],
                           [[0, 0, 1],
                            [1, 0, 0]]])
        self.img = np.array([[None, None], [None, None]])

        # Set up the plot
        self.fig, self.ax = pl.subplots(2, 2, figsize=(7, 7))
        for axis in self.ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)

        # Plot the initial image
        poly = y2p(self.y)
        for p in range(2):
            for q in range(2):
                flux = np.zeros((100, 100)) * np.nan
                for i, x in enumerate(np.linspace(-1, 1, 100)):
                    for j, y in enumerate(np.linspace(-1, 1, 100)):
                        ylim = np.sqrt(1 - x ** 2)
                        if y > -ylim and y < ylim:
                            flux[j][i] = evaluate_poly(poly, x, y)
                self.img[p, q] = self.ax[p, q].imshow(flux, cmap='plasma',
                                                      interpolation="none",
                                                      origin="lower",
                                                      extent=(-1, 1, -1, 1))

        # Set up the animation
        self.theta = np.linspace(0, 2 * np.pi, frames, endpoint=False)
        self.animation = animation.FuncAnimation(self.fig, self.animate,
                                                 frames=self.frames,
                                                 interval=50,
                                                 repeat=True, blit=True)

        # Save
        # self.animation.save('ylms.gif', writer='imagemagick',
        #                    fps=fps, dpi=dpi)
        # pl.close()
        pl.show()

    def animate(self, k):
        """Run the animation."""
        # print("Rendering frame %d/%d..." % (j + 1, self.frames))
        for p in range(2):
            for q in range(2):
                Ry = np.dot(R(2, self.u[p, q], self.theta[k]), self.y)
                poly = y2p(Ry)
                flux = np.zeros((100, 100)) * np.nan
                for i, x in enumerate(np.linspace(-1, 1, 100)):
                    for j, y in enumerate(np.linspace(-1, 1, 100)):
                        ylim = np.sqrt(1 - x ** 2)
                        if y > -ylim and y < ylim:
                            flux[j][i] = evaluate_poly(poly, x, y)
                self.img[p, q].set_data(flux)
        return self.img[0, 0], self.img[0, 1], self.img[1, 0], self.img[1, 1]


animated()
