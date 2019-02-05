import starry
import starry
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Marker size is proportional to log error
def ms(error):
    return 18 + np.log10(error)


npts = 1000
nsamples = 20
xo = np.linspace(-1.1, 1.1, npts)
lmax_arr = [0, 1, 2, 3, 4, 5, 10, 15]
t = np.zeros((2, 2, len(lmax_arr))) * np.nan
e = np.zeros((2, 2, len(lmax_arr))) * np.nan

for i, lmax in tqdm(enumerate(lmax_arr), total=len(lmax_arr)):
    map = starry.Map(lmax, multi=True)
    map[:, :] = 1
    truth = map.flux(xo=xo, yo=0.2, ro=0.1)
    for version, map in zip([0, 1], [starry.Map(lmax), starry.Map(lmax)]):
        map[:, :] = 1
        tk = np.zeros(nsamples) * np.nan
        for gradient in [False, True]:
            for k in range(nsamples):
                tstart = time.time()
                flux = map.flux(xo=xo, yo=0.2, ro=0.1, gradient=gradient)
                tk[k] = time.time() - tstart
            if gradient:
                flux = flux[0]
            e[version, int(gradient), i] = np.median(np.abs(flux - truth))
            t[version, int(gradient), i] = np.median(tk)


# Set up
fig = plt.figure(figsize=(8, 4))
ax = plt.subplot2grid((2, 5), (0, 0), colspan=4, rowspan=2)
axleg1 = plt.subplot2grid((2, 5), (0, 4))
axleg2 = plt.subplot2grid((2, 5), (1, 4))
axleg1.axis('off')
axleg2.axis('off')
ax.set_xlabel('Spherical harmonic degree', fontsize=12)
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
ax.set_ylabel('Evaluation time [seconds]', fontsize=12)
ax.set_yscale("log")

# Plot lines
ax.plot(lmax_arr, t[0, 0], color="C1", ls="-")
ax.plot(lmax_arr, t[0, 1], color="C1", ls="--")
ax.plot(lmax_arr, t[1, 0], color="C0", ls="-")
ax.plot(lmax_arr, t[1, 1], color="C0", ls="--")

# Plot points (size = error)
for i in range(len(lmax_arr)):
    ax.plot(lmax_arr[i], t[0, 0, i], color="C1", ls="none", 
            marker="o", ms=ms(e[0, 0, i]))
    ax.plot(lmax_arr[i], t[0, 1, i], color="C1", ls="none", 
            marker="o", ms=ms(e[0, 1, i]))
    ax.plot(lmax_arr[i], t[1, 0, i], color="C0", ls="none", 
            marker="o", ms=ms(e[1, 0, i]))
    ax.plot(lmax_arr[i], t[1, 1, i], color="C0", ls="none", 
            marker="o", ms=ms(e[1, 1, i]))

# Legend
axleg1.plot([0, 1], [0, 1], ls="-", color='C1', label='beta')
axleg1.plot([0, 1], [0, 1], ls="--", color='C1', label='beta (+grad)')
axleg1.plot([0, 1], [0, 1], ls="-", color='C0', label='v1.0')
axleg1.plot([0, 1], [0, 1], ls="--", color='C0', label='v1.0 (+grad)')
axleg1.set_xlim(2, 3)
leg = axleg1.legend(loc='center', frameon=False)
leg.set_title('version', prop={'weight': 'bold'})
for logerr in [-16, -12, -8, -4, 0]:
    axleg2.plot([0, 1], [0, 1], 'o', color='gray',
                ms=ms(10 ** logerr),
                label=r'$%3d$' % logerr)
axleg2.set_xlim(2, 3)
leg = axleg2.legend(loc='center', labelspacing=1, frameon=False)
leg.set_title('log error', prop={'weight': 'bold'})

# Save
fig.savefig("compare_to_beta.pdf", bbox_inches='tight')