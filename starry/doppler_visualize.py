from ._plotting import *
from bokeh.io import save, output_file, output_notebook, show
from bokeh.server.server import Server
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Span,
    Div,
    CheckboxButtonGroup,
    Band,
    Range1d,
    LinearAxis,
)
from bokeh.events import (
    Pan,
    Tap,
    MouseMove,
    MouseWheel,
    MouseEnter,
    MouseLeave,
)
from bokeh.palettes import Category20
from bokeh.resources import INLINE
import numpy as np
from IPython.display import HTML, display

PLOT_COUNTER = 0

HELP = """
Interact with the visualization by moving the mouse over
the map at the top left to show the emergent, rest frame spectrum at
different points on the surface in the plot at the bottom left.
Scroll (with the mouse wheel or track pad) to change the wavelength
at which the map is visualized (top left) or to rotate the orthographic
projection of the map (top right).
The plot at the bottom right shows the observed spectrum at the current
phase (black) and the corresponding rest frame spectrum absent Doppler
shifts (orange).
"""

HELP_ICON = """
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="30px"
	 height="30px" viewBox="0 0 30 30" style="enable-background:new 0 0 30 30;" xml:space="preserve">
<g id="Icons" style="opacity:0.75;">
	<g id="help">
		<path id="circle" style="fill-rule:evenodd;clip-rule:evenodd;" d="M12.001,2.085c-5.478,0-9.916,4.438-9.916,9.916
			c0,5.476,4.438,9.914,9.916,9.914c5.476,0,9.914-4.438,9.914-9.914C21.915,6.523,17.477,2.085,12.001,2.085z M12.002,20.085
			c-4.465,0-8.084-3.619-8.084-8.083c0-4.465,3.619-8.084,8.084-8.084c4.464,0,8.083,3.619,8.083,8.084
			C20.085,16.466,16.466,20.085,12.002,20.085z"/>
		<g id="question_mark">
			<path id="top" style="fill-rule:evenodd;clip-rule:evenodd;" d="M11.766,6.688c-2.5,0-3.219,2.188-3.219,2.188l1.411,0.854
				c0,0,0.298-0.791,0.901-1.229c0.516-0.375,1.625-0.625,2.219,0.125c0.701,0.885-0.17,1.587-1.078,2.719
				C11.047,12.531,11,15,11,15h1.969c0,0,0.135-2.318,1.041-3.381c0.603-0.707,1.443-1.338,1.443-2.494S14.266,6.688,11.766,6.688z"
				/>
			<rect id="bottom" x="11" y="16" style="fill-rule:evenodd;clip-rule:evenodd;" width="2" height="2"/>
		</g>
	</g>
</g>
<g id="Guides" style="display:none;">
</g>
</svg>
"""

SCRIPT = lambda counter: """
<style>
    .starry-help-modal {{
        display: none;
        position: relative;
        z-index: 99999;
        padding-top: 100px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgb(0,0,0);
        background-color: rgba(0,0,0,0.4);
        font-family: Arial, Helvetica, sans-serif;
    }}
    .starry-help-modal-content {{
        background-color: #fefefe;
        line-height: 1.4em;
        text-align: justify;
        width: 50% !important;
        margin: auto;
        padding: 20px;
        border: 1px solid #888;
        width: 80%;
    }}
    .starry-help-modal-content p {{
        margin: 30px;
    }}
    .starry-help-close {{
        color: #aaaaaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }}
    .starry-help-close:hover,
    .starry-help-close:focus {{
        color: #000;
        text-decoration: none;
        cursor: pointer;
    }}
    .starry-help-icon {{
        position: relative;
        top: 0px;
        left: 85%;
        font-size: 30px;
        opacity: 0.25;
        cursor: pointer;
        z-index: 99999;
    }}
    .starry-help-icon:hover,
    .starry-help-icon:focus {{
        opacity: 0.5;
    }}
</style>

<script>
    // Disable the natural page scrolling when
    // hovering over one of the plots
    var DISABLE_WHEEL = false;
    var disable_wheel = function (e) {{
        var plot = document.getElementsByClassName("plot_{counter:d}")[0];
        if (DISABLE_WHEEL && (plot != null)) e.preventDefault();
    }};
    window.addEventListener("wheel", disable_wheel, {{ passive:false }});

    // Add a help icon to the corner of the page
    // that opens a modal popup
    var plotLoaded = setInterval(function() {{
        var plot = document.getElementsByClassName("plot_{counter:d}")[0];
        var plot_ortho = document.getElementsByClassName("plot_ortho_{counter:d}")[0];
        if ((plot != null) && (plot_ortho != null)) {{
            clearInterval(plotLoaded);
            if (document.getElementsByClassName("starry-help-icon").length == 0) {{
                var help_modal = document.createElement("div");
                help_modal.classList.add("starry-help-modal");
                var help_modal_content = document.createElement("div");
                help_modal_content.classList.add("starry-help-modal-content");
                var help_close = document.createElement("span");
                help_close.classList.add("starry-help-close");
                help_close.innerHTML = "&times;";
                var help_text = document.createElement("p");
                help_text.innerHTML = "{help:s}";
                help_modal_content.appendChild(help_close);
                help_modal_content.appendChild(help_text);
                help_modal.appendChild(help_modal_content);
                plot.appendChild(help_modal);
                var help_icon = document.createElement("div");
                help_icon.innerHTML = '{icon:s}';
                help_icon.classList.add("starry-help-icon");
                plot_ortho.appendChild(help_icon);
                help_icon.onclick = function() {{
                    help_modal.style.display = "block";
                }}
                help_close.onclick = function() {{
                    help_modal.style.display = "none";
                }}
            }}
        }}
    }}, 100);
</script>
""".format(
    help=HELP.replace("\n", " "),
    icon=HELP_ICON.replace("\n", " "),
    counter=counter,
)

TEMPLATE = lambda counter: """
{{% block postamble %}}
<style>
.bk-root .bk {{
    margin: 0 auto !important;
}}
</style>
{:s}
{{% endblock %}}
""".format(
    SCRIPT(counter)
)


class Visualize:
    def __init__(
        self, wavs, wavf, moll, ortho, spec, theta, flux0, flux, inc, **kwargs
    ):
        # Current plot counter
        global PLOT_COUNTER
        self.counter = PLOT_COUNTER
        PLOT_COUNTER += 1

        # Store as single precision
        self.wavs = np.array(wavs, dtype="float32")
        self.wavf = np.array(wavf, dtype="float32")
        self.moll = np.array(moll, dtype="float32")
        self.ortho = np.array(ortho, dtype="float32")
        self.spec = np.array(spec, dtype="float32")
        self.flux0 = np.array(flux0, dtype="float32")
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
        mx = np.nanmax(values)
        mn = np.nanmin(values)
        rg = max(1e-3, mx - mn)
        self.vmax_m = mx + 0.1 * rg
        self.vmin_m = mn - 0.1 * rg
        mx = np.nanmax(self.ortho)
        mn = np.nanmin(self.ortho)
        rg = max(1e-3, mx - mn)
        self.vmax_o = mx + 0.1 * rg
        self.vmin_o = mn - 0.1 * rg
        mx = np.nanmax(self.flux)
        mn = np.nanmin(self.flux)
        rg = max(1e-3, mx - mn)
        self.vmax_f = mx + 0.1 * rg
        self.vmin_f = mn - 0.1 * rg
        mx = np.nanmax(self.flux0)
        mn = np.nanmin(self.flux0)
        rg = max(1e-3, mx - mn)
        self.vmax_f0 = mx + 0.1 * rg
        self.vmin_f0 = mn - 0.1 * rg

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
            data=dict(flux=self.flux[0], flux0=self.flux0[0], wavf=self.wavf)
        )
        self.source_index = ColumnDataSource(
            data=dict(l=[self.nws // 2], t=[0])
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
        self.spec_vline = Span(
            location=self.wavs[self.nws // 2],
            dimension="height",
            line_color=Category20[3][2],
            line_width=3,
            line_alpha=0.5,
        )

    def add_border(self, plot, projection="ortho", pts=1000):
        if projection == "ortho":
            xe = np.linspace(-1, 1, pts)
            ye = np.sqrt(1 - xe ** 2)
            res = self.npix_o
        else:
            xe = np.linspace(-2, 2, pts)
            ye = 0.5 * np.sqrt(4 - xe ** 2)
            res = self.npix_m
        d = 1 - 2 * np.sqrt(2) / res

        # Fill above
        source = ColumnDataSource(
            data=dict(x=d * xe, lower=d * ye, upper=np.ones_like(xe))
        )
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            level="overlay",
            fill_alpha=1,
            fill_color="white",
            line_width=0,
            source=source,
        )
        plot.add_layout(band)

        # Fill below
        source = ColumnDataSource(
            data=dict(x=d * xe, lower=-np.ones_like(xe), upper=-d * ye)
        )
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            level="overlay",
            fill_alpha=1,
            fill_color="white",
            line_width=0,
            source=source,
        )
        plot.add_layout(band)

        # Fill right
        if projection == "ortho":
            source = ColumnDataSource(
                data=dict(
                    x=[1 + 1.1 * (d - 1), 1.1],
                    lower=[-1.0, -1.0],
                    upper=[1.0, 1.0],
                )
            )
        else:
            source = ColumnDataSource(
                data=dict(
                    x=[2 * (1 + 1.1 * (d - 1)), 2.1],
                    lower=[-1.0, -1.0],
                    upper=[1.0, 1.0],
                )
            )
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            level="overlay",
            fill_alpha=1.0,
            fill_color="white",
            line_width=0,
            source=source,
        )
        plot.add_layout(band)

        # Fill left
        if projection == "ortho":
            source = ColumnDataSource(
                data=dict(
                    x=[-1.1, -(1 + 1.1 * (d - 1))],
                    lower=[-1.0, -1.0],
                    upper=[1.0, 1.0],
                )
            )
        else:
            source = ColumnDataSource(
                data=dict(
                    x=[-2.1, -2 * (1 + 1.1 * (d - 1))],
                    lower=[-1.0, -1.0],
                    upper=[1.0, 1.0],
                )
            )
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            level="overlay",
            fill_alpha=1.0,
            fill_color="white",
            line_width=0,
            source=source,
        )
        plot.add_layout(band)

        # Plot contour
        plot.line(d * xe, d * ye, line_width=2, color="black", alpha=1)
        plot.line(d * xe, -d * ye, line_width=2, color="black", alpha=1)

    def plot_moll(self):
        # Plot the map
        plot_moll = figure(
            aspect_ratio=2,
            toolbar_location=None,
            x_range=(-2, 2),
            y_range=(-1, 1),
            id="plot_moll",
            name="plot_moll",
        )
        plot_moll.axis.visible = False
        plot_moll.grid.visible = False
        plot_moll.outline_line_color = None
        color_mapper = LinearColorMapper(
            palette="Plasma256",
            nan_color="white",
            low=self.vmin_m,
            high=self.vmax_m,
        )
        plot_moll.image(
            image="moll",
            x=-2,
            y=-1,
            dw=4,
            dh=2,
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
        self.add_border(plot_moll, "moll")

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
                "spec_vline": self.spec_vline,
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
                source_index.change.emit();
                spec_vline.location = wavs[l];

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

        mouse_enter_callback = CustomJS(
            code="""
            DISABLE_WHEEL= true;
            """
        )
        plot_moll.js_on_event(MouseEnter, mouse_enter_callback)

        mouse_leave_callback = CustomJS(
            code="""
            DISABLE_WHEEL = false;
            """
        )
        plot_moll.js_on_event(MouseLeave, mouse_leave_callback)

        return plot_moll

    def plot_spec(self):
        # Plot the spectrum
        plot_spec = figure(
            plot_width=280,
            plot_height=130,
            toolbar_location=None,
            x_range=(self.wavs[0], self.wavs[-1]),
            y_range=(self.vmin_m, self.vmax_m),
            min_border_left=40,
            min_border_right=40,
        )
        plot_spec.line(
            "wavs",
            "spec",
            source=self.source_spec,
            line_width=1,
            color="black",
        )
        plot_spec.renderers.extend([self.spec_vline])

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

        plot_spec.yaxis.axis_label = "local intensity"
        plot_spec.yaxis.axis_label_text_color = "black"
        plot_spec.yaxis.axis_label_standoff = 10
        plot_spec.yaxis.axis_label_text_font_style = "normal"

        plot_spec.outline_line_width = 1.5
        plot_spec.outline_line_alpha = 1
        plot_spec.outline_line_color = "black"

        return plot_spec

    def plot_ortho(self):
        # Plot the map
        plot_ortho = figure(
            aspect_ratio=2,
            toolbar_location=None,
            x_range=(-2, 2),
            y_range=(-1, 1),
            id="plot_ortho",
            name="plot_ortho",
            min_border_left=0,
            min_border_right=0,
            css_classes=["plot_ortho_{:d}".format(self.counter)],
        )
        plot_ortho.axis.visible = False
        plot_ortho.grid.visible = False
        plot_ortho.outline_line_color = None
        color_mapper = LinearColorMapper(
            palette="Plasma256",
            nan_color="white",
            low=self.vmin_o,
            high=self.vmax_o,
        )
        plot_ortho.image(
            image="ortho",
            x=-1,
            y=-1,
            dw=2,
            dh=2,
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
        self.add_border(plot_ortho, "ortho")

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
                "flux0": self.flux0,
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
                while (tidx < 0) tidx += nt;
                while (tidx > nt - 1) tidx -= nt;

                // Update the map
                source_ortho.data["ortho"][0] = ortho[tidx];
                source_ortho.change.emit();

                // Update the longitude lines
                var k;
                for (var k = 0; k < nlon; k++) {
                    source_ortho_lon.data["x" + k] = lon_x[tidx][k];
                    source_ortho_lon.data["y" + k] = lon_y[tidx][k];
                }
                source_ortho_lon.change.emit();

                // Update the flux
                source_flux.data["flux"] = flux[tidx];
                source_flux.data["flux0"] = flux0[tidx];
                source_flux.change.emit();
                """,
        )
        plot_ortho.js_on_event(MouseWheel, mouse_wheel_callback)

        mouse_enter_callback = CustomJS(
            code="""
            DISABLE_WHEEL = true;
            """
        )
        plot_ortho.js_on_event(MouseEnter, mouse_enter_callback)

        mouse_leave_callback = CustomJS(
            code="""
            DISABLE_WHEEL = false;
            """
        )
        plot_ortho.js_on_event(MouseLeave, mouse_leave_callback)

        return plot_ortho

    def plot_flux(self):
        # Plot the flux (the output spectrum)
        plot_flux = figure(
            plot_width=280,
            plot_height=130,
            toolbar_location=None,
            x_range=(self.wavf[0], self.wavf[-1]),
            y_range=(self.vmin_f, self.vmax_f),
            min_border_left=40,
            min_border_right=40,
        )

        # Rest frame
        plot_flux.extra_y_ranges["flux0"] = Range1d(
            start=self.vmin_f0, end=self.vmax_f0
        )
        plot_flux.add_layout(LinearAxis(y_range_name="flux0"), "right")
        plot_flux.line(
            "wavf",
            "flux0",
            source=self.source_flux,
            line_width=1,
            color=Category20[3][2],
            alpha=0.5,
            y_range_name="flux0",
        )

        # Observed
        plot_flux.line(
            "wavf",
            "flux",
            source=self.source_flux,
            line_width=1,
            color="black",
        )

        plot_flux.yaxis[0].axis_label = "observed intensity"
        plot_flux.yaxis[0].axis_label_text_color = "black"
        plot_flux.yaxis[0].axis_label_standoff = 10
        plot_flux.yaxis[0].axis_label_text_font_style = "normal"

        plot_flux.yaxis[1].axis_label = "rest intensity"
        plot_flux.yaxis[1].axis_label_text_color = Category20[3][2]
        plot_flux.yaxis[1].axis_label_text_alpha = 0.5
        plot_flux.yaxis[1].axis_label_standoff = 10
        plot_flux.yaxis[1].axis_label_text_font_style = "normal"
        plot_flux.yaxis[1].major_label_text_color = Category20[3][2]
        plot_flux.yaxis[1].major_label_text_alpha = 0.5

        plot_flux.toolbar.active_drag = None
        plot_flux.toolbar.active_scroll = None
        plot_flux.toolbar.active_tap = None
        plot_flux.xgrid.grid_line_color = None
        plot_flux.ygrid.grid_line_color = None

        plot_flux.xaxis.axis_label = "wavelength (nm)"
        plot_flux.xaxis.axis_label_text_color = "black"
        plot_flux.xaxis.axis_label_standoff = 10
        plot_flux.xaxis.axis_label_text_font_style = "normal"

        plot_flux.outline_line_width = 1.5
        plot_flux.outline_line_alpha = 1
        plot_flux.outline_line_color = "black"

        return plot_flux

    def layout(self):
        return row(
            column(
                self.plot_moll(),
                column(self.plot_spec(), sizing_mode="scale_width"),
                sizing_mode="scale_width",
            ),
            Div(),
            column(
                self.plot_ortho(),
                column(self.plot_flux(), sizing_mode="scale_width"),
                sizing_mode="scale_width",
            ),
            min_width=600,
            max_width=1200,
            css_classes=["plot_{:d}".format(self.counter)],
        )

    def show_notebook(self):
        # Define the function we'll use to disable mouse wheel
        # scrolling when hovering over a plot
        display(HTML(SCRIPT(self.counter)))
        output_notebook(hide_banner=True)
        show(self.layout())

    def save(self, file="starry.html"):
        save(
            self.layout(),
            filename=file,
            title="starry",
            template=TEMPLATE(self.counter),
            resources=INLINE,
        )

    def _launch(self, doc):
        doc.title = "starry"
        doc.template = TEMPLATE(self.counter)
        doc.add_root(self.layout())

    def launch(self):
        server = Server({"/": self._launch})
        server.start()
        print("Press Ctrl+C to exit.")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def show(self, file=None):
        if file is not None:
            assert file.endswith(".htm") or file.endswith(
                ".html"
            ), "Keyword `file` must be a path to an HTML file."
            self.save(file=file)
        else:
            # Are we in a Jupyter notebook?
            try:
                if "zmqshell" in str(type(get_ipython())):
                    # YES: display inline
                    self.show_notebook()
                else:
                    raise NameError("")
            except NameError:
                self.launch()
