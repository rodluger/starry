from ._plotting import *
from bokeh.io import save, output_file
from bokeh.server.server import Server
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColumnDataSource, CustomJS, Span, Div
from bokeh.events import Pan, Tap, MouseMove, MouseWheel
from bokeh.palettes import Category20
import numpy as np


class Visualize:
    def __init__(
        self, wavs, wavf, moll, ortho, spec, theta, flux, inc, **kwargs
    ):
        # Store as single precision
        self.wavs = np.array(wavs, dtype="float32")
        self.wavf = np.array(wavf, dtype="float32")
        self.moll = np.array(moll, dtype="float32")
        self.ortho = np.array(ortho, dtype="float32")
        self.spec = np.array(spec, dtype="float32")
        self.flux = np.array(flux, dtype="float32")
        self.theta = np.array(theta, dtype="float32")
        self.inc = inc

        # Dimensions
        self.nc = self.moll.shape[0]
        self.npix_m = self.moll.shape[1]
        self.npix_o = self.ortho.shape[1]
        self.nws = self.spec.shape[1]
        self.nwf = self.flux.shape[1]
        self.nt = self.ortho.shape[0]

        # Get plot ranges
        values = (
            self.spec.T @ self.moll[:, ::10, ::10].reshape(self.nc, -1)
        ).flatten()
        self.vmax_m = 1.1 * np.nanmax(values)
        self.vmax_o = 1.1 * np.nanmax(self.ortho)
        self.vmax_f = 1.1 * np.nanmax(self.flux)

        # Get the image at the central wavelength bin
        moll0 = (
            self.spec[:, self.nws // 2] @ self.moll.reshape(self.nc, -1)
        ).reshape(self.npix_m, self.npix_m)

        # Get the spectrum at the center of the image
        spec0 = self.spec.T @ self.moll[:, self.npix_m // 2, self.npix_m // 2]

        # Data sources
        self.source_moll = ColumnDataSource(data=dict(moll=[moll0]))
        self.source_ortho = ColumnDataSource(data=dict(ortho=[self.ortho[0]]))
        self.source_spec = ColumnDataSource(
            data=dict(spec=spec0, wavs=self.wavs)
        )
        self.source_flux = ColumnDataSource(
            data=dict(flux=self.flux[0], wavf=self.wavf)
        )
        self.source_index = ColumnDataSource(
            data=dict(
                l=[self.nws // 2], x=[self.wavs[self.nws // 2]], y=[0.0], t=[0]
            )
        )
        lon_lines = []
        for m in range(self.nt):
            lon_lines.append(
                get_ortho_longitude_lines(inc=self.inc, theta=self.theta[m])
            )
        lon_lines = np.array(lon_lines, dtype="float32")
        self.lon_x, self.lon_y = np.swapaxes(
            np.swapaxes(lon_lines, 0, 2), 1, 2
        )
        data_dict = {}
        for i, x, y in zip(
            range(len(self.lon_x[0])), self.lon_x[0], self.lon_y[0]
        ):
            data_dict["x{:d}".format(i)] = x
            data_dict["y{:d}".format(i)] = y
        self.source_ortho_lon = ColumnDataSource(data=data_dict)

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
            x=-2 - epsp,
            y=-1 - 0.5 * epsp,
            dw=4 + 2 * epsp,
            dh=2 + epsp,
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

        # Hack to hide pixel edges
        xe = np.linspace(-2, 2, 1000)
        ye = 0.5 * np.sqrt(4 - xe ** 2)
        xe *= 1.015
        ye *= 1.06
        plot_moll.line(xe, ye, line_width=10, color="white", alpha=1)
        plot_moll.line(xe, -ye, line_width=10, color="white", alpha=1)

        # Actual map border
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
                "nws": self.nws,
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
                        var local_spec = new Array(nws).fill(0);
                        for (var k = 0; k < nc; k++) {
                            var weight = moll[k][j][i];
                            for (var l = 0; l < nws; l++) {
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
                "nws": self.nws,
            },
            code="""
                // Update the current wavelength index
                var delta = Math.floor(cb_obj["delta"]);
                var l = source_index.data["l"][0];
                l += delta;
                if (l < 0) l = 0;
                if (l > nws - 1) l = nws - 1;
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
            x=-1 - epsp,
            y=-1 - epsp,
            dw=2 + 2 * epsp,
            dh=2 + 2 * epsp,
            color_mapper=color_mapper,
            source=self.source_ortho,
        )
        plot_ortho.toolbar.active_drag = None
        plot_ortho.toolbar.active_scroll = None
        plot_ortho.toolbar.active_tap = None

        # Plot the lat/lon grid
        lat_lines = get_ortho_latitude_lines(inc=self.inc)
        for x, y in lat_lines:
            plot_ortho.line(x, y, line_width=1, color="black", alpha=0.25)
        for i in range(len(self.lon_x[0])):
            plot_ortho.line(
                "x{:d}".format(i),
                "y{:d}".format(i),
                line_width=1,
                color="black",
                alpha=0.25,
                source=self.source_ortho_lon,
            )

        # Hack to hide pixel edges
        xe = np.linspace(-1, 1, 1000)
        ye = np.sqrt(1 - xe ** 2)
        xe *= 1.03
        ye *= 1.03
        plot_ortho.line(xe, ye, line_width=10, color="white", alpha=1)
        plot_ortho.line(xe, -ye, line_width=10, color="white", alpha=1)

        # Actual map border
        xe = np.linspace(-1, 1, 1000)
        ye = np.sqrt(1 - xe ** 2)
        plot_ortho.line(xe, ye, line_width=3, color="black", alpha=1)
        plot_ortho.line(xe, -ye, line_width=3, color="black", alpha=1)

        # Interaction: Rotate the star as the mouse wheel moves
        mouse_wheel_callback = CustomJS(
            args={
                "source_ortho": self.source_ortho,
                "source_index": self.source_index,
                "source_flux": self.source_flux,
                "source_ortho_lon": self.source_ortho_lon,
                "lon_x": self.lon_x,
                "lon_y": self.lon_y,
                "nlon": len(self.lon_x[0]),
                "ortho": self.ortho,
                "flux": self.flux,
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
                var tidx = Math.floor(t);

                // Update the map
                source_ortho.data["ortho"][0] = ortho[tidx];
                source_ortho.change.emit();

                // Update the longitude lines
                for (var n = 0; n < nlon; n++) {
                    source_ortho_lon.data["x" + n] = lon_x[tidx][n];
                    source_ortho_lon.data["y" + n] = lon_y[tidx][n];
                }
                source_ortho_lon.change.emit();

                // Update the flux
                source_flux.data["flux"] = flux[tidx];
                source_flux.change.emit();
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
            x_range=(self.wavf[0], self.wavf[-1]),
            y_range=(0, self.vmax_f),
        )
        plot_flux.line(
            "wavf",
            "flux",
            source=self.source_flux,
            line_width=1,
            color="black",
        )

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

    def save(self, file="starry.html"):
        template = """
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
        output_file(file, title="starry")
        save(layout, filename=file, title="starry", template=template)

    def _launch(self, doc):
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

    def launch(self):
        server = Server({"/": self._launch})
        server.start()
        print("Press Ctrl+C to exit.")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
