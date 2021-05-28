from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColumnDataSource, CustomJS, Span
from bokeh.events import Pan, Tap, MouseMove, MouseWheel
from bokeh.palettes import Category20
import numpy as np
import starry


def get_latitude_lines(dlat=np.pi / 6, npts=1000, niter=100):
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:
        theta = lat
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = np.linspace(-2, 2, npts)
        y = np.ones(npts) * np.sin(theta)
        a = 1
        b = 2
        y[(y / a) ** 2 + (x / b) ** 2 > 1] = np.nan
        res.append((x, y))
    return res


def get_longitude_lines(dlon=np.pi / 6, npts=1000, niter=100):
    res = []
    lonlines = np.arange(-np.pi, np.pi, dlon)[1:]
    for lon in lonlines:
        lat = np.linspace(-np.pi / 2, np.pi / 2, npts)
        theta = np.array(lat)
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = 2 / np.pi * lon * np.cos(theta)
        y = np.sin(theta)
        res.append((x, y))
    return res


class Visualize:
    def __init__(self, wavs, image, spec):

        # Dimensions
        nc = image.shape[0]
        npix = image.shape[1]
        nw = spec.shape[1]

        # Get plot ranges
        values = (spec.T @ image[:, ::10, ::10].reshape(nc, -1)).flatten()
        vmax = 1.1 * np.nanmax(values)

        # Get the image at the central wavelength bin
        image0 = (spec[:, nw // 2] @ image.reshape(nc, -1)).reshape(npix, npix)

        # Get the spectrum at the center of the image
        spec0 = spec.T @ image[:, npix // 2, npix // 2]

        # Data sources
        source_image = ColumnDataSource(data=dict(image=[image0]))
        source_spec = ColumnDataSource(data=dict(spec=spec0, wavs=wavs))
        source_index = ColumnDataSource(
            data=dict(l=[nw // 2], x=[wavs[nw // 2]], y=[0.0])
        )

        # Plot the map
        eps = 0.1
        epsp = 0.01
        plot_image = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(-2 - eps, 2 + eps),
            y_range=(-1 - eps / 2, 1 + eps / 2),
        )
        plot_image.axis.visible = False
        plot_image.grid.visible = False
        plot_image.outline_line_color = None
        color_mapper = LinearColorMapper(
            palette="Plasma256", nan_color="white", low=0.0, high=vmax
        )
        plot_image.image(
            image="image",
            x=-2,
            y=-1,
            dw=4,
            dh=2 + epsp / 2,
            color_mapper=color_mapper,
            source=source_image,
        )
        plot_image.toolbar.active_drag = None
        plot_image.toolbar.active_scroll = None
        plot_image.toolbar.active_tap = None

        # Plot the lat/lon grid
        lat_lines = get_latitude_lines()
        lon_lines = get_longitude_lines()
        for x, y in lat_lines:
            plot_image.line(x, y, line_width=1, color="black", alpha=0.25)
        for x, y in lon_lines:
            plot_image.line(x, y, line_width=1, color="black", alpha=0.25)
        xe = np.linspace(-2, 2, 1000)
        ye = 0.5 * np.sqrt(4 - xe ** 2)
        plot_image.line(xe, ye, line_width=3, color="black", alpha=1)
        plot_image.line(xe, -ye, line_width=3, color="black", alpha=1)

        # Plot the spectrum
        plot_spec = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(wavs[0], wavs[-1]),
            y_range=(0, vmax),
        )
        plot_spec.line(
            "wavs", "spec", source=source_spec, line_width=1, color="black"
        )
        plot_spec.ray(
            x="x",
            y="y",
            length=300,
            angle=0.5 * np.pi,
            source=source_index,
            line_width=3,
            color=Category20[3][2],
            alpha=0.5,
        )
        plot_spec.toolbar.active_drag = None
        plot_spec.toolbar.active_scroll = None
        plot_spec.toolbar.active_tap = None

        # Interaction
        mouse_move_callback = CustomJS(
            args={
                "source_spec": source_spec,
                "image": image,
                "spec": spec,
                "npix": npix,
                "nc": nc,
                "nw": nw,
            },
            code="""
                var x = cb_obj["x"];
                var y = cb_obj["y"];

                if ((x > - 2) && (x < 2) && (y > -1) && (y < 1)) {

                    // Image index below cursor
                    var i = Math.floor(0.25 * (x + 2) * npix);
                    var j = Math.floor(0.5 * (y + 1) * npix);

                    // Compute weighted spectrum
                    if (!isNaN(image[0][j][i])) {
                        var local_spec = new Array(nw).fill(0);
                        for (var k = 0; k < nc; k++) {
                            var weight = image[k][j][i];
                            for (var l = 0; l < nw; l++) {
                                local_spec[l] += weight * spec[k][l]
                            }
                        }

                        // Update the plot
                        source_spec.data["spec"] = local_spec;
                        source_spec.change.emit();
                    }
                }
                """,
        )
        plot_image.js_on_event(MouseMove, mouse_move_callback)

        mouse_wheel_callback = CustomJS(
            args={
                "source_image": source_image,
                "source_index": source_index,
                "image": image,
                "spec": spec,
                "wavs": wavs,
                "npix": npix,
                "nc": nc,
                "nw": nw,
            },
            code="""
                // Update the current wavelength index
                var delta = cb_obj["delta"];
                var l = source_index.data["l"][0];
                l += delta;
                if (l < 0) l = 0;
                if (l > nw - 1) l = nw - 1;
                source_index.data["l"][0] = l;
                source_index.data["x"][0] = wavs[l];
                source_index.change.emit();

                // Update the map
                var local_image = new Array(npix).fill(0).map(() => new Array(npix).fill(0));
                for (var k = 0; k < nc; k++) {
                    var weight = spec[k][l];
                    for (var i = 0; i < npix; i++) {
                        for (var j = 0; j < npix; j++) {
                            local_image[j][i] += weight * image[k][j][i];
                        }
                    }
                }
                source_image.data["image"][0] = local_image;
                source_image.change.emit();
                """,
        )
        plot_image.js_on_event(MouseWheel, mouse_wheel_callback)

        # Full layout
        self.layout = column(plot_image, plot_spec)

    def run(self, doc):
        doc.title = "starry"
        doc.template = """
        {% block postamble %}
        <style>
        .bk-root .bk {
            margin: 0 auto !important;
        }
        </style>
        <script>
            window.addEventListener("wheel", e => e.preventDefault(), { passive:false });
        </script>
        {% endblock %}
        """
        doc.add_root(self.layout)


wavs = np.linspace(642.5, 643.5, 199)
image = np.zeros((4, 500, 500))
map = starry.Map(20, lazy=False)
for k, letter in enumerate(["s", "p", "o", "t"]):
    map.load(letter, force_psd=True)
    img = map.render(projection="moll", res=500)
    image[k] = img / np.nanmax(img)
spec = 1.0 - np.array(
    [
        np.exp(-0.5 * (wavs - wavs[len(wavs) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(2 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(3 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(4 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
    ]
)

viz = Visualize(wavs, image, spec)

server = Server({"/": viz.run})
server.start()
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
