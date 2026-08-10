import fastf1
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from scipy import signal
import os

# Real Lap you want to compare with
YEAR = 2026
TRACK = 'Montreal'
SESSION = 'Q'

# Sim Data File
SIM_FILE = f'Results/sim_telemetry_{TRACK}.npz'

BG         = '#12141a'
PANEL_BG   = '#181b22'
GRID_C     = '#333844'
EDGE_C     = '#454b56'
TEXT_C     = '#e8e9ec'
COL_SIM    = '#3ddc84'  # Green for the 2026 Sim
COL_REAL   = '#5fa8ff'  # Blue for Real Telemetry
COL_DELTA  = '#ffb454'  # Orange for the Delta


print(f"Loading FastF1 Telemetry: {YEAR} {TRACK} {SESSION}...")
session = fastf1.get_session(YEAR, TRACK, SESSION)
session.load(telemetry=True, weather=False, messages=False)

# Get the absolute fastest lap of the session 
fastest_lap = session.laps.pick_fastest()
driver = fastest_lap['Driver']
real_tel = fastest_lap.get_telemetry()

s_real = real_tel['Distance'].to_numpy()
v_real = real_tel['Speed'].to_numpy()

print(f"Loading Sim Telemetry from {SIM_FILE}...")
if not os.path.exists(SIM_FILE):
    raise FileNotFoundError(f"Could not find {SIM_FILE}. Did you export it from the optimizer?")

sim_data = np.load(SIM_FILE)
s_sim = sim_data['s']
v_sim = sim_data['v_kmh']

min_len = min(len(s_sim), len(v_sim))
s_sim = s_sim[:min_len]
v_sim = v_sim[:min_len]

# Normalize both distances to a 0.0 -> 1.0 scale
s_real_norm = (s_real - s_real[0]) / (s_real[-1] - s_real[0])
s_sim_norm = (s_sim - s_sim[0]) / (s_sim[-1] - s_sim[0])

# Interpolate the Real telemetry onto the Sim's exact nodes
interp_function = PchipInterpolator(s_real_norm, v_real)
v_real_aligned = interp_function(s_sim_norm)

# Find the exact phase shift (offset) between the two speed traces
correlation = signal.correlate(v_sim - np.mean(v_sim), v_real_aligned - np.mean(v_real_aligned), mode='full')
lags = signal.correlation_lags(len(v_sim), len(v_real_aligned), mode='full')

# Find the lag that produces the highest correlation overlap
best_lag = lags[np.argmax(correlation)]
print(f"Auto-Aligning traces: Shifting real telemetry by {best_lag} nodes.")

# Shift the real telemetry to perfectly lock the braking zones together
v_real_aligned = np.roll(v_real_aligned, best_lag)

# Calculate the Speed Delta
speed_delta = v_sim - v_real_aligned


fig, (ax_speed, ax_delta) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.patch.set_facecolor(BG)
fig.suptitle(f"Telemetry Correlation: 2026 Sim vs {YEAR} Real ({driver})", color=TEXT_C, fontsize=16, fontweight='bold')

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_C, linestyle='--', alpha=0.7)
    ax.tick_params(colors=TEXT_C)
    for spine in ax.spines.values():
        spine.set_color(EDGE_C)

style_ax(ax_speed)
style_ax(ax_delta)


ax_speed.plot(s_sim_norm * 100, v_real_aligned, color=COL_REAL, linewidth=2.5, label=f'Real F1 ({YEAR} {driver})', alpha=0.8)
ax_speed.plot(s_sim_norm * 100, v_sim, color=COL_SIM, linewidth=2.5, label='2026 CasADi Sim', alpha=0.9)
ax_speed.set_ylabel('Speed [km/h]', color=TEXT_C, fontsize=12)
ax_speed.legend(loc='upper right', frameon=False, labelcolor=TEXT_C, fontsize=11)

ax_delta.axhline(0, color=TEXT_C, linewidth=1, alpha=0.5)
ax_delta.fill_between(s_sim_norm * 100, 0, speed_delta, where=(speed_delta >= 0), color='#3ddc84', alpha=0.4, label='Sim Faster')
ax_delta.fill_between(s_sim_norm * 100, 0, speed_delta, where=(speed_delta < 0), color='#ff5c5c', alpha=0.4, label='Real Faster')
ax_delta.plot(s_sim_norm * 100, speed_delta, color=COL_DELTA, linewidth=1.5)

ax_delta.set_xlabel('Lap Distance [%]', color=TEXT_C, fontsize=12)
ax_delta.set_ylabel('Δ Speed [km/h]', color=TEXT_C, fontsize=12)
ax_delta.legend(loc='upper right', frameon=False, labelcolor=TEXT_C, fontsize=10)
ax_delta.set_xlim(0, 100)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
