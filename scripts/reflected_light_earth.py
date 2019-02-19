import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import starry


# Plot the disk
fig, ax = plt.subplots(1, figsize=(8, 7))
x = np.linspace(-1, 1, 10000)
y = np.sqrt(1 - x ** 2)
ax.plot(x, +y, 'k-', lw=3)
ax.plot(x, -y, 'k-', lw=3)
ax.set_aspect(1)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis('off')

# Vector to source
vec, = plt.plot([], [], 'k--', alpha=0.5, lw=1)

# Source image
src, = plt.plot([], [], "k*", ms=10)

# Sub-source point
sub, = plt.plot([], [], "k.", ms=10)

# Terminator
term, = plt.plot([], [], "k-")

# The `starry` map
map = starry.Map(lmax=20, reflected=True)
map.load_image("earth")
npts = 159
vmin = 0.0
vmax = np.nanmax(map.render(source=[0,0,1], res=npts))
alpha = 1.0
Z = np.zeros((npts, npts)) * np.nan
pts = plt.imshow(Z, origin="lower", 
                 extent=(-1, 1, -1, 1), 
                 vmin=vmin, vmax=vmax,
                 cmap="plasma", interpolation="none")

# User sliders
axt = plt.axes([0.15, 0.20, 0.7, 0.03])
axx = plt.axes([0.15, 0.15, 0.7, 0.03])
axy = plt.axes([0.15, 0.10, 0.7, 0.03])
axz = plt.axes([0.15, 0.05, 0.7, 0.03])
st = Slider(axt, r'$\theta$', -180.0, 180.0, valinit=0, valstep=1.0)
sx = Slider(axx, 'x', -1.0, 1.0, valinit=np.sqrt(1./3.), valstep=0.01)
sy = Slider(axy, 'y', -1.0, 1.0, valinit=np.sqrt(1./3.), valstep=0.01)
sz = Slider(axz, 'z', -1.0, 1.0, valinit=np.sqrt(1./3.), valstep=0.01)

# Light curve
axl = plt.axes([0.15, 0.80, 0.7, 0.1])
axl.axis('off')
flux = np.array([map.flux(source=[1, 1, 1])])
time = np.array([0.0])
lc, = axl.plot(flux, time, lw=1)

# Interactors
no_update = False
def update(val):
    # Hack to prevent infinite recursion
    global no_update, time, flux
    if no_update:
        return

    # Normalize the source vector
    s = np.array([sx.val, sy.val, sz.val])
    s /= np.sqrt(np.sum(s ** 2))
    no_update = True
    sx.set_val(s[0])
    sy.set_val(s[1])
    sz.set_val(s[2])
    no_update = False

    # Plot vector to source
    vec.set_xdata([s[0], 2 * s[0]])
    vec.set_ydata([s[1], 2 * s[1]])
    src.set_xdata([2 * s[0]])
    src.set_ydata([2 * s[1]])

    # Plot sub-source point
    if s[2] < 0:
        sub.set_alpha(0.1)
        if (4 * (s[0] ** 2 + s[1] ** 2) < 1):
            src.set_alpha(0.1)
        else:
            src.set_alpha(1)
    else:
        sub.set_alpha(1)
        src.set_alpha(1)
    sub.set_xdata([s[0]])
    sub.set_ydata([s[1]])

    # Update terminator ellipse
    b = -s[2]
    y = b * np.sqrt(1 - x ** 2)

    # Rotate terminator
    r = np.sqrt(sx.val ** 2 + sy.val ** 2)
    cosw = sy.val / r
    sinw = -sx.val / r
    xt = x * cosw - y * sinw
    yt = x * sinw + y * cosw
    term.set_xdata(xt)
    term.set_ydata(yt)

    # Update map
    Z = map.render(theta=st.val, source=s, res=npts)
    pts.set_data(Z)

    # Update light curve
    time = np.append(time, [time[-1] + 1])
    flux = np.append(flux, [map.flux(source=s, theta=st.val)])
    lc.set_data(time, flux)
    axl.set_xlim(0, time[-1])
    axl.set_ylim(0.75 * np.min(flux), 1.25 * np.max(flux))

    # Draw
    fig.canvas.draw_idle()

st.on_changed(update)
sx.on_changed(update)
sy.on_changed(update)
sz.on_changed(update)
update(0)

# Show
plt.show()