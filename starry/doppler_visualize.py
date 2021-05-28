from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColumnDataSource, CustomJS, Span, Div
from bokeh.events import Pan, Tap, MouseMove, MouseWheel
from bokeh.palettes import Category20
import numpy as np

# TODO: Make relative
import starry
from starry._plotting import *


class Visualize:
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, wavs, moll, ortho, spec, inc, **kwargs):
        # Store
        self.wavs = wavs
        self.moll = moll
        self.ortho = ortho
        self.spec = spec
        self.inc = inc

        # Dimensions
        self.nc = moll.shape[0]
        self.npix_m = moll.shape[1]
        self.npix_o = ortho.shape[1]
        self.nw = spec.shape[1]
        self.nt = ortho.shape[0]

        # Get plot ranges
        values = (spec.T @ moll[:, ::10, ::10].reshape(self.nc, -1)).flatten()
        self.vmax_m = 1.1 * np.nanmax(values)
        self.vmax_o = 1.1 * np.nanmax(ortho)

        # Get the image at the central wavelength bin
        moll0 = (spec[:, self.nw // 2] @ moll.reshape(self.nc, -1)).reshape(
            self.npix_m, self.npix_m
        )

        # Get the spectrum at the center of the image
        spec0 = spec.T @ moll[:, self.npix_m // 2, self.npix_m // 2]

        # Data sources
        self.source_moll = ColumnDataSource(data=dict(moll=[moll0]))
        self.source_ortho = ColumnDataSource(data=dict(ortho=[ortho[0]]))
        self.source_spec = ColumnDataSource(data=dict(spec=spec0, wavs=wavs))
        self.source_index = ColumnDataSource(
            data=dict(l=[self.nw // 2], x=[wavs[self.nw // 2]], y=[0.0], t=[0])
        )

    def plot_moll(self):
        # Plot the map
        eps = 0.1
        epsp = 0.01
        plot_moll = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(-2 - eps, 2 + eps),
            y_range=(-1 - eps / 2, 1 + eps / 2),
        )
        plot_moll.axis.visible = False
        plot_moll.grid.visible = False
        plot_moll.outline_line_color = None
        color_mapper = LinearColorMapper(
            palette="Plasma256", nan_color="white", low=0.0, high=self.vmax_m
        )
        plot_moll.image(
            image="moll",
            x=-2,
            y=-1,
            dw=4,
            dh=2 + epsp / 2,
            color_mapper=color_mapper,
            source=self.source_moll,
        )
        plot_moll.toolbar.active_drag = None
        plot_moll.toolbar.active_scroll = None
        plot_moll.toolbar.active_tap = None

        # Plot the lat/lon grid
        lat_lines = get_moll_latitude_lines()
        lon_lines = get_moll_longitude_lines()
        for x, y in lat_lines:
            plot_moll.line(
                x / np.sqrt(2),
                y / np.sqrt(2),
                line_width=1,
                color="black",
                alpha=0.25,
            )
        for x, y in lon_lines:
            plot_moll.line(
                x / np.sqrt(2),
                y / np.sqrt(2),
                line_width=1,
                color="black",
                alpha=0.25,
            )
        xe = np.linspace(-2, 2, 1000)
        ye = 0.5 * np.sqrt(4 - xe ** 2)
        plot_moll.line(xe, ye, line_width=3, color="black", alpha=1)
        plot_moll.line(xe, -ye, line_width=3, color="black", alpha=1)

        # Interaction: show spectra at different points as mouse moves
        mouse_move_callback = CustomJS(
            args={
                "source_spec": self.source_spec,
                "moll": self.moll,
                "spec": self.spec,
                "npix_m": self.npix_m,
                "nc": self.nc,
                "nw": self.nw,
            },
            code="""
                var x = cb_obj["x"];
                var y = cb_obj["y"];

                if ((x > - 2) && (x < 2) && (y > -1) && (y < 1)) {

                    // Image index below cursor
                    var i = Math.floor(0.25 * (x + 2) * npix_m);
                    var j = Math.floor(0.5 * (y + 1) * npix_m);

                    // Compute weighted spectrum
                    if (!isNaN(moll[0][j][i])) {
                        var local_spec = new Array(nw).fill(0);
                        for (var k = 0; k < nc; k++) {
                            var weight = moll[k][j][i];
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
        plot_moll.js_on_event(MouseMove, mouse_move_callback)

        # Interaction: Cycle through wavelength as mouse wheel moves
        mouse_wheel_callback = CustomJS(
            args={
                "source_moll": self.source_moll,
                "source_index": self.source_index,
                "moll": self.moll,
                "spec": self.spec,
                "wavs": self.wavs,
                "npix_m": self.npix_m,
                "nc": self.nc,
                "nw": self.nw,
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
                var local_moll = new Array(npix_m).fill(0).map(() => new Array(npix_m).fill(0));
                for (var k = 0; k < nc; k++) {
                    var weight = spec[k][l];
                    for (var i = 0; i < npix_m; i++) {
                        for (var j = 0; j < npix_m; j++) {
                            local_moll[j][i] += weight * moll[k][j][i];
                        }
                    }
                }
                source_moll.data["moll"][0] = local_moll;
                source_moll.change.emit();
                """,
        )
        plot_moll.js_on_event(MouseWheel, mouse_wheel_callback)

        return plot_moll

    def plot_spec(self):
        # Plot the spectrum
        plot_spec = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(self.wavs[0], self.wavs[-1]),
            y_range=(0, self.vmax_m),
        )
        plot_spec.line(
            "wavs",
            "spec",
            source=self.source_spec,
            line_width=1,
            color="black",
        )
        plot_spec.ray(
            x="x",
            y="y",
            length=300,
            angle=0.5 * np.pi,
            source=self.source_index,
            line_width=3,
            color=Category20[3][2],
            alpha=0.5,
        )
        plot_spec.toolbar.active_drag = None
        plot_spec.toolbar.active_scroll = None
        plot_spec.toolbar.active_tap = None

        plot_spec.toolbar.active_drag = None
        plot_spec.toolbar.active_scroll = None
        plot_spec.toolbar.active_tap = None
        plot_spec.xgrid.grid_line_color = None
        plot_spec.ygrid.grid_line_color = None

        plot_spec.xaxis.axis_label = "wavelength (nm)"
        plot_spec.xaxis.axis_label_text_color = "black"
        plot_spec.xaxis.axis_label_standoff = 10
        plot_spec.xaxis.axis_label_text_font_style = "normal"

        plot_spec.yaxis.axis_label = "local spectral intensity"
        plot_spec.yaxis.axis_label_text_color = "black"
        plot_spec.yaxis.axis_label_standoff = 10
        plot_spec.yaxis.axis_label_text_font_style = "normal"

        plot_spec.outline_line_width = 1.5
        plot_spec.outline_line_alpha = 1
        plot_spec.outline_line_color = "black"

        return plot_spec

    def plot_ortho(self):
        # Plot the map
        eps = 0.05
        epsp = 0.005
        plot_ortho = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(-270 / 130 - eps, 270 / 130 + eps),
            y_range=(-1 - eps, 1 + eps),
        )
        plot_ortho.axis.visible = False
        plot_ortho.grid.visible = False
        plot_ortho.outline_line_color = None
        color_mapper = LinearColorMapper(
            palette="Plasma256", nan_color="white", low=0.0, high=self.vmax_o
        )
        plot_ortho.image(
            image="ortho",
            x=-1,
            y=-1,
            dw=2,
            dh=2 + epsp,
            color_mapper=color_mapper,
            source=self.source_ortho,
        )
        plot_ortho.toolbar.active_drag = None
        plot_ortho.toolbar.active_scroll = None
        plot_ortho.toolbar.active_tap = None

        # Plot the lat/lon grid
        lat_lines = get_ortho_latitude_lines(inc=self.inc * np.pi / 180)
        lon_lines = get_ortho_longitude_lines(inc=self.inc * np.pi / 180)
        for x, y in lat_lines:
            plot_ortho.line(x, y, line_width=1, color="black", alpha=0.25)
        for x, y in lon_lines:
            plot_ortho.line(x, y, line_width=1, color="black", alpha=0.25)
        xe = np.linspace(-1, 1, 1000)
        ye = np.sqrt(1 - xe ** 2)
        plot_ortho.line(xe, ye, line_width=3, color="black", alpha=1)
        plot_ortho.line(xe, -ye, line_width=3, color="black", alpha=1)

        # Interaction: Rotate the star as the mouse wheel moves
        mouse_wheel_callback = CustomJS(
            args={
                "source_ortho": self.source_ortho,
                "source_index": self.source_index,
                "ortho": self.ortho,
                "npix_o": self.npix_o,
                "nt": self.nt,
                "speed": self.nt / 200,
            },
            code="""
                // Update the current theta index
                var delta = cb_obj["delta"];
                var t = source_index.data["t"][0];
                t += delta * speed;
                while (t < 0) t += nt;
                while (t > nt - 1) t -= nt;
                source_index.data["t"][0] = t;
                source_index.change.emit();

                // Update the map
                source_ortho.data["ortho"][0] = ortho[Math.floor(t)];
                source_ortho.change.emit();
                """,
        )
        plot_ortho.js_on_event(MouseWheel, mouse_wheel_callback)

        return plot_ortho

    def plot_flux(self):
        # Plot the flux (the output spectrum)
        plot_flux = figure(
            plot_width=2 * 280,
            plot_height=2 * 130,
            toolbar_location=None,
            x_range=(self.wavs[0], self.wavs[-1]),
            y_range=(0, self.vmax_m),
        )

        # TODO

        plot_flux.toolbar.active_drag = None
        plot_flux.toolbar.active_scroll = None
        plot_flux.toolbar.active_tap = None
        plot_flux.xgrid.grid_line_color = None
        plot_flux.ygrid.grid_line_color = None

        plot_flux.xaxis.axis_label = "wavelength (nm)"
        plot_flux.xaxis.axis_label_text_color = "black"
        plot_flux.xaxis.axis_label_standoff = 10
        plot_flux.xaxis.axis_label_text_font_style = "normal"

        plot_flux.yaxis.axis_label = "observed spectral intensity"
        plot_flux.yaxis.axis_label_text_color = "black"
        plot_flux.yaxis.axis_label_standoff = 10
        plot_flux.yaxis.axis_label_text_font_style = "normal"

        plot_flux.outline_line_width = 1.5
        plot_flux.outline_line_alpha = 1
        plot_flux.outline_line_color = "black"

        return plot_flux

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
            // Disable mouse wheel on page
            window.addEventListener("wheel", e => e.preventDefault(), { passive:false });
        </script>
        {% endblock %}
        """
        layout = row(
            column(self.plot_moll(), self.plot_spec()),
            Div(),
            column(self.plot_ortho(), self.plot_flux()),
        )
        doc.add_root(layout)


# Sample map
wavs = np.linspace(642.5, 643.5, 199)
moll = np.zeros((4, 500, 500))
ortho = np.zeros((8, 150, 150))
theta = np.linspace(0, 360, 8, endpoint=False)
inc = 60
map = starry.Map(20, inc=inc, lazy=False)
for k, letter in enumerate(["s", "p", "o", "t"]):
    map.load(letter, force_psd=True)
    img = map.render(projection="moll", res=500)
    moll[k] = img / np.nanmax(img)
    ortho += map.render(projection="ortho", theta=theta, res=150)
spec = 1.0 - np.array(
    [
        np.exp(-0.5 * (wavs - wavs[len(wavs) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(2 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(3 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
        np.exp(-0.5 * (wavs - wavs[(4 * len(wavs)) // 5]) ** 2 / 0.05 ** 2),
    ]
)
viz = Visualize(wavs, moll, ortho, spec, inc)

# Launch
server = Server({"/": viz.run})
server.start()
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
