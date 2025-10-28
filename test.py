import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

np.random.seed(7)

months = np.arange(0, 25)
real = 44 + months / 12.0  # real age increases

# Bio-age: start 41 -> peak ~48 at m18 -> ease to ~47 at m24
rise = 41 + (48 - 41) * (months / 18) ** 0.9
fall = 48 - (48 - 47) * np.clip((months - 18) / 6, 0, 1) ** 1.2
bio = np.where(months <= 18, rise, fall)
bio = np.clip(bio + np.random.normal(0, 0.12, bio.shape), 40, 50)
bio[0] = 41.0

# Segment colors (dark palette)
c_green = '#14532d'
c_amber = '#b45309'
c_red   = '#991b1b'
c_real  = '#94a3b8'
bg      = '#0b1220'
grid    = '#334155'
txt     = '#cbd5e1'
spine   = '#64748b'

# Continuous, color-coded line
pts = np.column_stack([months, bio])
segs = np.stack([pts[:-1], pts[1:]], axis=1)
mid_diff = (bio[:-1] + bio[1:]) / 2 - (real[:-1] + real[1:]) / 2
seg_colors = np.where(mid_diff <= 0, c_green, np.where(mid_diff < 2, c_amber, c_red))

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(2.7, 2.9), dpi=100, constrained_layout=True)
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# Area shading vs dynamic real age
ax.fill_between(months, bio, real, where=bio <= real, color=c_green, alpha=0.35, interpolate=True)
ax.fill_between(months, real, np.minimum(bio, real + 2), where=bio > real, color=c_amber, alpha=0.35, interpolate=True)
ax.fill_between(months, real + 2, bio, where=bio >= real + 2, color=c_red, alpha=0.35, interpolate=True)

# Add continuous colored line and real age line
lc = LineCollection(segs, colors=seg_colors, linewidths=2.6, linestyles='solid', zorder=3)
lc.set_capstyle('round'); lc.set_joinstyle('round')
ax.add_collection(lc)
ax.plot(months, real, color=c_real, ls='--', lw=1.3, label='Real age')

# Compact styling
ax.set(title='Biological Age (2y)', xlabel='Months', ylabel='Age (years)',
       xlim=(0, 24), ylim=(40, 50))
ax.set_xticks([0, 12, 24]); ax.set_yticks([40, 45, 50])
ax.tick_params(colors=txt, labelsize=8, length=3)
for s in ['top','right','left','bottom']:
    ax.spines[s].set_color(spine)
ax.xaxis.label.set_color(txt); ax.yaxis.label.set_color(txt); ax.title.set_color(txt)
ax.grid(alpha=0.25, zorder=0, color=grid)

# Legend
ax.plot([], [], color=c_green, lw=3, label='≤ Real age')
ax.plot([], [], color=c_amber, lw=3, label='< 2 years')
ax.plot([], [], color=c_red, lw=3, label='≥ 2 years')
ax.legend(frameon=False, loc='upper left', fontsize=8)

plt.savefig('/home/ubuntu/datathon/bio_age2.png', dpi=300)
plt.show()