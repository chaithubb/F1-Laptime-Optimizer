import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import fastf1

#Extract Elevation Data from FastF1
TRACK = 'Imola'
F1_YEAR = 2025 

#Input CSV Coordinates and Active Aero Zones
CSV_FILE = f'Circuit Data/Raw Coordinates/{TRACK}.csv'
OUTPUT_FILE = f'Circuit Data/Processed Circuit/{TRACK}_processed_e.npz'
AERO_ZONES_FILE = f'Circuit Data/Active Aero Zones/{TRACK}_aero_zones.csv'
N = 1000  
SMOOTHING_FACTOR = 0.1

print(f"Loading track geometry from {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: Could not find '{CSV_FILE}'.")
    exit()

x_raw = df['x_m'].values
y_raw = df['y_m'].values
w_right_raw = df['w_tr_right_m'].values
w_left_raw = df['w_tr_left_m'].values

if np.hypot(x_raw[0] - x_raw[-1], y_raw[0] - y_raw[-1]) > 0.1:
    x_raw = np.append(x_raw, x_raw[0])
    y_raw = np.append(y_raw, y_raw[0])
    w_right_raw = np.append(w_right_raw, w_right_raw[0])
    w_left_raw = np.append(w_left_raw, w_left_raw[0])


print("Fitting periodic B-Splines to centerline...")
tck, u_raw = splprep([x_raw, y_raw], s=SMOOTHING_FACTOR, per=1)

u_fine = np.linspace(0, 1, 10000)
x_fine, y_fine = splev(u_fine, tck)
dx_f, dy_f = splev(u_fine, tck, der=1)
ddx_f, ddy_f = splev(u_fine, tck, der=2)

kappa_fine = (dx_f * ddy_f - dy_f * ddx_f) / ((dx_f**2 + dy_f**2)**1.5)
ds_fine_step = np.hypot(np.diff(x_fine), np.diff(y_fine))
ds_fine_step = np.append(ds_fine_step, ds_fine_step[-1])
s_fine = np.concatenate(([0], np.cumsum(ds_fine_step[:-1])))
total_distance = s_fine[-1]

print(f"Applying Adaptive Mesh Refinement (Budget: {N} Nodes)...")

alpha = 100.0 
density = 1.0 + alpha * np.abs(kappa_fine)

tau = np.cumsum(density * ds_fine_step)
tau_norm = tau / tau[-1]

tau_even = np.linspace(0, 1, N + 1)[:-1]

s_adaptive = np.interp(tau_even, tau_norm, s_fine)
u_adaptive = np.interp(s_adaptive, s_fine, u_fine)

x_spline, y_spline = splev(u_adaptive, tck)
dx, dy = splev(u_adaptive, tck, der=1)
ddx, ddy = splev(u_adaptive, tck, der=2)

kappa = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
psi_unwrapped = np.unwrap(np.arctan2(dy, dx))

s_next = np.roll(s_adaptive, -1)
ds_array = s_next - s_adaptive
ds_array[-1] = total_distance - s_adaptive[-1]
ds_array[ds_array < 0] += total_distance 

s_raw_approx = np.concatenate(([0], np.cumsum(np.hypot(np.diff(x_raw), np.diff(y_raw)))))
cs_right = CubicSpline(s_raw_approx, w_right_raw, bc_type='periodic')
cs_left = CubicSpline(s_raw_approx, w_left_raw, bc_type='periodic')
w_right_interp = cs_right(s_adaptive)
w_left_interp = cs_left(s_adaptive)

u_aero = np.zeros(N)
try:
    df_aero = pd.read_csv(AERO_ZONES_FILE)
    print(f"Applying Active Aero zones from {AERO_ZONES_FILE}...")
    
    for index, row in df_aero.iterrows():
        start = row['start_m']
        end = row['end_m']
        mode = row['mode']
        
        if end == -1:
            end = total_distance
        
        zone_indices = np.where((s_adaptive >= start) & (s_adaptive <= end))[0]
        if len(zone_indices) > 0:
            u_aero[zone_indices] = mode
            
except FileNotFoundError:
    print(f"Notice: '{AERO_ZONES_FILE}' not found. Proceeding with NO active aero zones.")
except Exception as e:
    print(f"Warning: Could not parse '{AERO_ZONES_FILE}'. Error: {e}")

print(f"\nFetching FastF1 telemetry for {TRACK} to extract elevation...")
try:
    session = fastf1.get_session(F1_YEAR, TRACK, 'Q')
    session.load(telemetry=True, weather=False, messages=False)
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    f1_x = tel['X'].values / 10.0 # Convert to meters
    f1_y = tel['Y'].values / 10.0
    f1_z = tel['Z'].values / 10.0 

    # 1. Clean duplicates based on distance
    _, unique_indices = np.unique(tel['Distance'].values, return_index=True)
    f1_x_clean = f1_x[unique_indices]
    f1_y_clean = f1_y[unique_indices]
    f1_z_clean = f1_z[unique_indices]

    # 1b. PRE-SMOOTHING: Remove GPS jitter
    z_window = 10 
    f1_z_smooth = pd.Series(f1_z_clean).rolling(window=z_window, min_periods=1, center=True).mean().values
    f1_z_smooth[-1] = f1_z_smooth[0]

    # 2. SPATIAL ALIGNMENT (Bounding Box Centering)
    # We cannot use Kabsch here because the arrays have different lengths and no point-to-point correspondence.
    # Instead, we assume both are North-Up and align their bounding box centers.
    f1_pts = np.column_stack((f1_x_clean, f1_y_clean))
    custom_pts = np.column_stack((x_spline, y_spline))
    
    f1_center = (np.max(f1_pts, axis=0) + np.min(f1_pts, axis=0)) / 2.0
    custom_center = (np.max(custom_pts, axis=0) + np.min(custom_pts, axis=0)) / 2.0
    
    f1_pts_centered = f1_pts - f1_center
    custom_pts_centered = custom_pts - custom_center

    # 3. SPATIAL MATCHING (KDTree)
    tree = KDTree(f1_pts_centered)
    distances, indices = tree.query(custom_pts_centered)

    # Map the Z values based on closest physical coordinate
    custom_z_raw = f1_z_smooth[indices]
    
    # 4. POST-SMOOTHING with periodic boundaries to remove snapping noise
    window_size = 20
    padded_z = np.concatenate([custom_z_raw[-window_size:], custom_z_raw, custom_z_raw[:window_size]])
    smoothed_z = pd.Series(padded_z).rolling(window=window_size, center=True).mean().values
    custom_z = smoothed_z[window_size:-window_size]
    custom_z[-1] = custom_z[0] # Ensure perfect closure
    
    # 5. Calculate Theta (slope angle) using REAL distances
    dz_ds = np.gradient(custom_z, s_adaptive)
    theta_raw = np.arctan(dz_ds)

    # 6. Smooth theta periodically
    padded_theta = np.concatenate([theta_raw[-15:], theta_raw, theta_raw[:15]])
    smoothed_theta = pd.Series(padded_theta).rolling(window=15, center=True).mean().values
    theta = smoothed_theta[15:-15]
    theta[-1] = theta[0]

except Exception as e:
    print(f"Warning: FastF1 elevation sync failed: {e}. Defaulting to flat track.")
    theta = np.zeros(N)
    custom_z = np.zeros(N)

print(f"Total Distance:        {total_distance:.1f} m")
print(f"Nodes (N):             {N}")
print(f"Node Spacing (Min):    {np.min(ds_array):.2f} m (In Tight Corners)")
print(f"Node Spacing (Max):    {np.max(ds_array):.2f} m (On Long Straights)")

np.savez_compressed(OUTPUT_FILE, N=N, L_track=total_distance, 
                    kappa=kappa, 
                    w_tr_right=w_right_interp, 
                    w_tr_left=w_left_interp,
                    ds_array=ds_array,
                    s_track=s_adaptive,
                    u_aero=u_aero,
                    theta=theta) 
print(f" Success! Saved to: {OUTPUT_FILE}\n")


plt.figure(figsize=(10, 8))
plt.title(f'Adaptive Node Placement (N={N})', fontsize=16, fontweight='bold')
plt.plot(x_spline, y_spline, 'k-', linewidth=1.0, alpha=0.5, label='Centerline')
plt.scatter(x_spline, y_spline, c=kappa, cmap='coolwarm', s=15, zorder=5, label='Adaptive Nodes')

if np.any(u_aero != 0):
    offset_aero = w_right_interp + 3.0
    x_aero = x_spline + offset_aero * np.sin(psi_unwrapped)
    y_aero = y_spline - offset_aero * np.cos(psi_unwrapped)

    aero_points = np.array([x_aero, y_aero]).T.reshape(-1, 1, 2)
    aero_segments = np.concatenate([aero_points[:-1], aero_points[1:]], axis=1)
    active_idx = np.where(u_aero[:-1] != 0)[0]

    if active_idx.size > 0:
        from matplotlib.collections import LineCollection
        lc_aero = LineCollection(aero_segments[active_idx], colors='magenta', linewidths=4, alpha=0.8, zorder=6)
        plt.gca().add_collection(lc_aero)

    plt.plot([], [], color='magenta', linewidth=4, alpha=0.8, label='Active Aero Zone')

plt.axis('equal')
plt.xlabel('X Coordinate [m]')
plt.ylabel('Y Coordinate [m]')
cbar = plt.colorbar()
cbar.set_label('Curvature Intensity', rotation=270, labelpad=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.figure(figsize=(10, 8))
plt.title('Track Elevation Map', fontsize=16, fontweight='bold')
plt.plot(x_spline, y_spline, 'k-', linewidth=1.0, alpha=0.5, label='Centerline')
plt.scatter(x_spline, y_spline, c=custom_z, cmap='viridis', s=15, zorder=5, label='Elevation Nodes')
plt.axis('equal')
plt.xlabel('X Coordinate [m]')
plt.ylabel('Y Coordinate [m]')
cbar_elev = plt.colorbar()
cbar_elev.set_label('Elevation (Z) [m]', rotation=270, labelpad=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.figure(figsize=(12, 4))
plt.title(r'Node Spacing ($ds$) vs Distance', fontsize=14, fontweight='bold')
plt.plot(s_adaptive, ds_array, color='darkorange', linewidth=2, label='Distance to Next Node')
plt.xlabel('Distance around track (s) [m]', fontsize=12)
plt.ylabel('Node Spacing (ds) [m]', fontsize=12)
plt.axhline(np.mean(ds_array), color='black', linestyle='--', label='Average Spacing')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.figure(figsize=(12, 4))
plt.title('Track Curvature (Kappa) vs Distance', fontsize=14, fontweight='bold')
plt.plot(s_adaptive, kappa, color='purple', linewidth=1.5, label='Curvature (k)')
plt.fill_between(s_adaptive, 0, kappa, color='purple', alpha=0.2)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5) 
plt.xlabel('Distance around track (s) [m]', fontsize=12)
plt.ylabel('Curvature [1/m]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.set_title('Track Elevation & Gradient (Theta)', fontsize=14, fontweight='bold')
ax1.plot(s_adaptive, custom_z, color='green', linewidth=2, label='Elevation [m]')
ax1.set_xlabel('Distance around track (s) [m]', fontsize=12)
ax1.set_ylabel('Elevation (Z) [m]', color='green', fontsize=12)
ax1.tick_params(axis='y', labelcolor='green')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
theta_deg = np.degrees(theta)
ax2.plot(s_adaptive, theta_deg, color='blue', linewidth=1.5, alpha=0.6, label='Slope Angle [deg]')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_ylabel('Slope Angle (Theta) [degrees]', color='blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='blue')

fig.tight_layout()
plt.show()