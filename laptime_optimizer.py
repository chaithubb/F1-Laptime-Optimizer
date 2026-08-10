import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os

#USES MA57 Solver which requires Academic Licence, Remove if not accessible - Line 571

# Track Details
track = 'Imola'
track_file = f'Circuit Data/Processed Circuit/{track}_processed_e.npz'
E_regen_max_quali = 7.0 #Max Regen per lap in MJ
E_regen_max_race = 8.5
E_regen_max_OT = 9.0
R_deploy_ramp_down = 1.0 #Slew Rate in 100kW/s (0.5 for 50kW/s)
v_init = 77.0 #Quali Starting Speed in m/s
rho = 1.225 #Air Density in kg/m^3

# OPTIMIZER MODE (Quali Lap vs Race Lap vs Race Lap with OT Mode)
MODE = 'QUALIFYING' #QUALIFYING / RACE / RACE_OT

#PU REGs 
YEAR = '2026' 
if YEAR == '2026':
    P_ICE_max = 4.0   #ICE and MGUK power in 100kW
    P_MGUK_max_deploy = 3.5
    P_MGUK_max_regen = 3.5
    SoC_max = 4.0 #Battery Capacity in MJ

if YEAR == '2027':
    P_ICE_max = 4.2   #ICE and MGUK power in 100kW
    P_MGUK_max_deploy = 3.0
    P_MGUK_max_regen = 3.75
    SoC_max = 4.5 #Battery Capacity in MJ

if YEAR == '2028':
    P_ICE_max = 4.5  #ICE and MGUK power in 100kW
    P_MGUK_max_deploy = 3.0
    P_MGUK_max_regen = 4.0
    SoC_max = 5.0 #Battery Capacity in MJ

#SMOOTHING HELPERS to protect the Hessian from kinks
def smooth_pos(x):
    return 0.5 * (x + ca.sqrt(x**2 + 1e-4))

def smooth_floor(x, floor_val):
    diff = x - floor_val
    return 0.5 * (diff + ca.sqrt(diff**2 + 1e-4)) + floor_val

def smooth_min(a, b):
    return 0.5 * (a + b - ca.sqrt((a - b)**2 + 1e-4))

# 0. GLOBAL PLOT THEME 
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'axes.titleweight': 'bold',
})
 
BG         = '#12141a'   
PANEL_BG   = '#181b22'   
TARMAC     = '#23262e'   
GRID_C     = '#333844'
EDGE_C     = '#454b56'
TEXT_C     = '#e8e9ec'
MUTED_C    = '#9aa0ab'
 
COL_PRIMARY  = '#e8e9ec'   
COL_DEPLOY   = "#6b05f9"   
COL_REGEN    = '#3ddc84'   
COL_NEUTRAL  = "#aeafb1"   
COL_AERO     = "#f012a2"   
COL_SOC      = '#c792ea'   
COL_CAR      = '#ffd54a'   
COL_ICE      = '#ffa94d'   
COL_BRAKE_F  = '#ff5c5c'   
COL_BRAKE_R  = '#c93b52'   
COL_TRACTIVE = '#3ddc84'   
COL_FRONT    = COL_NEUTRAL 
COL_REAR     = COL_AERO    
COL_REGEN_BRAKE = COL_REGEN          
COL_PART_REGEN  = "#e88e0f"          
COL_SUPERCLIP   = '#c93b52'          
 
def style_axis(ax, xlabel=None, ylabel=None, legend_loc=None):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_C, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.tick_params(colors=MUTED_C, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(EDGE_C)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_C, fontsize=10.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_C, fontsize=10.5)
    if legend_loc:
        ax.legend(loc=legend_loc, frameon=False, labelcolor=TEXT_C, fontsize=9.5)
 
def style_figure(fig, title, subtitle=None):
    fig.patch.set_facecolor(BG)
    full_title = f"{title}   \u2014   {subtitle}" if subtitle else title
    fig.suptitle(full_title, fontsize=14.5, fontweight='bold', color=TEXT_C) 

# 1. INITIALIZATION & ADAPTIVE TRACK DATA
if os.path.exists(track_file):
    print("Loading real F1 track geometry...")
    track_data = np.load(track_file)
    N = int(track_data['N'])
    L_track = float(track_data['L_track'])
    kappa_ref = track_data['kappa']
    w_tr_right = track_data['w_tr_right']
    w_tr_left = track_data['w_tr_left']
    
    if 'theta' in track_data:
        theta = track_data['theta']
    else:
        theta = np.zeros(N) 
    
    if 'ds_array' in track_data:
        ds_array = track_data['ds_array']
        s_track = track_data['s_track']
    else:
        ds_array = np.full(N, L_track / N)
        s_track = np.linspace(0, L_track, N)
        
    if 'u_aero' in track_data:
        u_aero = track_data['u_aero']
    else:
        u_aero = np.zeros(N) 
        
# 2. VEHICLE & FRENET PARAMETERS
m = 778.0           
g = 9.81                    
Crr = 0.015         

L_base = 3.40       
wd_f = 0.45         
l_r = L_base * wd_f 
l_f = L_base * (1.0 - wd_f) 
h_cog = 0.28        
aero_bal_f = 0.45   

Fz0_f = m * g * wd_f         
Fz0_r = m * g * (1.0 - wd_f) 

p_dy1_long = 1.87   
p_dy1_lat = 1.88    
p_dy2 = -0.17       

CdA_Z = 1.62; ClA_Z = 3.62    
Delta_CdA = 0.42; Delta_ClA = 1.08 
      
SoC_min = 0.0       
F_brake_max = 30000.0 

R_wheel = 0.35
Final_Drive = 4.0

RPM_peak = 8600.0
T_peak = P_ICE_max * 95.5

gear_ratios = [3.25, 2.62, 2.02, 1.82, 1.53, 1.32, 1.20, 1.08]
shift_speeds = [25.0, 36.0, 44.0, 56.7, 65.3, 73.6, 83.3]
             
v_int = ca.MX.sym('v_int')
gr_smooth = gear_ratios[0]

for i in range(len(shift_speeds)):
    transition = 0.5 * (1.0 + ca.tanh(5.0 * (v_int - shift_speeds[i])))
    gr_smooth = gr_smooth + transition * (gear_ratios[i+1] - gear_ratios[i])

gear_discrete_fn = ca.Function('gear_discrete_fn', [v_int], [gr_smooth])

kW100_to_W = 100000.0
MW_to_W = 1000000.0

# 3. OPTIMIZER SETUP 

opti = ca.Opti()

#SCALING FACTORS 
V_SCALE = 100.0            
TIME_SCALE = 100.0         
SOC_SCALE = SoC_max        
E_REGEN_SCALE = E_regen_max_OT 
N_SCALE = 10.0             
XI_SCALE = 1.0             
P_ICE_SCALE = P_ICE_max    
P_MGUK_SCALE = P_MGUK_max_regen  
F_BRAKE_SCALE = F_brake_max 
DELTA_SCALE = 0.6          

X_s = opti.variable(6, N + 1)
U_s = opti.variable(5, N)

# The Slack Variable (to give a bit of leeway for the optimizer)
sigma_tire = opti.variable(N) 
opti.subject_to(sigma_tire >= 0.0) 
opti.set_initial(sigma_tire, 0.05)

t = X_s[0, :] * TIME_SCALE
n = X_s[1, :] * N_SCALE          
xi = X_s[2, :] * XI_SCALE         
v = X_s[3, :] * V_SCALE          
soc = X_s[4, :] * SOC_SCALE
e_regen = X_s[5, :] * E_REGEN_SCALE

P_ICE = U_s[0, :] * P_ICE_SCALE
P_MGUK = U_s[1, :] * P_MGUK_SCALE
F_brake_f = U_s[2, :] * F_BRAKE_SCALE
F_brake_r = U_s[3, :] * F_BRAKE_SCALE
delta = U_s[4, :] * DELTA_SCALE      

x_sym = ca.MX.sym('x', 6)
u_sym = ca.MX.sym('u', 5)
p_sym = ca.MX.sym('p', 3) 

n_sym = x_sym[1]
xi_sym = x_sym[2]
v_sym = x_sym[3]

P_ICE_W = u_sym[0] * kW100_to_W
P_MGUK_W = u_sym[1] * kW100_to_W
F_brake_f_sym = u_sym[2]
F_brake_r_sym = u_sym[3]
delta_sym = u_sym[4]

kappa_ref_sym = p_sym[1]
theta_sym = p_sym[0]

kappa_car_sym = ca.tan(delta_sym) / L_base
geom_factor = 1.0 - (n_sym * kappa_ref_sym)

dt_ds = geom_factor / (v_sym * ca.cos(xi_sym))
dn_ds = geom_factor * ca.tan(xi_sym)
dxi_ds = kappa_car_sym * (geom_factor / ca.cos(xi_sym)) - kappa_ref_sym


u_aero_base = p_sym[2]

#ICE Power available
gr_sym = gear_discrete_fn(v_sym)
omega_eng_sym = (v_sym / R_wheel) * gr_sym * Final_Drive
rpm_sym = omega_eng_sym * (30.0 / np.pi)
x_rpm_sym = rpm_sym / RPM_peak
t_curve_sym = T_peak * ((2.0 * x_rpm_sym) - x_rpm_sym**2)
P_ICE_avail_sym = smooth_min((t_curve_sym * omega_eng_sym) / kW100_to_W, P_ICE_max)

P_ICE_100kW = u_sym[0] 

brake_viol_sym = smooth_pos(((F_brake_f_sym + F_brake_r_sym) / F_brake_max) - 0.05)
aero_shutoff_sym = smooth_min(1.0, 100.0 * (brake_viol_sym)) #Active Aero shuts when braking
u_aero_dyn = u_aero_base * (1.0 - aero_shutoff_sym)

CdA_dyn = CdA_Z - (u_aero_dyn * Delta_CdA) #Drag coefficient when active aero is shut off vs on
ClA_dyn = ClA_Z - (u_aero_dyn * Delta_ClA) #Lift coefficient when active aero is shut off vs on

F_drag = 0.5 * rho * CdA_dyn * v_sym**2
F_df = 0.5 * rho * ClA_dyn * v_sym**2
F_norm = m * g * ca.cos(theta_sym) + F_df

F_pt = (P_ICE_W + 0.97 * P_MGUK_W) / v_sym #97% deployment efficiency for the MGUK
a_long = (F_pt - F_drag - (Crr * F_norm) - (m * g * ca.sin(theta_sym)) - F_brake_f_sym - F_brake_r_sym) / m

dv_ds = a_long * dt_ds

P_MGUK_MW = P_MGUK_W / MW_to_W
P_regen_MW_smooth = (ca.sqrt(P_MGUK_MW**2 + 1e-4) - P_MGUK_MW) / 2.0 #Regenerative braking power
P_deploy_MW_smooth = (ca.sqrt(P_MGUK_MW**2 + 1e-4) + P_MGUK_MW) / 2.0 #Power deployment from the battery

dsoc_ds = (( -P_deploy_MW_smooth) + (0.92 * P_regen_MW_smooth)) * dt_ds #92% regen efficiency
deregen_ds = (0.92 * P_regen_MW_smooth) * dt_ds

dx_ds = ca.vertcat(dt_ds, dn_ds, dxi_ds, dv_ds, dsoc_ds, deregen_ds)
f_physics = ca.Function('f_physics', [x_sym, u_sym, p_sym], [dx_ds])


# 5. DYNAMICS LOOP (HERMITE-SIMPSON + FOH)

scale_vector = ca.vertcat(TIME_SCALE, N_SCALE, XI_SCALE, V_SCALE, SOC_SCALE, E_REGEN_SCALE)

for k in range(N):

    x_k = ca.vertcat(t[k], n[k], xi[k], v[k], soc[k], e_regen[k])
    x_next = ca.vertcat(t[k+1], n[k+1], xi[k+1], v[k+1], soc[k+1], e_regen[k+1])
    
    u_k = ca.vertcat(P_ICE[k], P_MGUK[k], F_brake_f[k], F_brake_r[k], delta[k])
    
    k_next = (k + 1) % N
    u_next = ca.vertcat(P_ICE[k_next], P_MGUK[k_next], F_brake_f[k_next], F_brake_r[k_next], delta[k_next])
    
    u_c = 0.5 * (u_k + u_next)
    
    p_k = ca.vertcat(theta[k], kappa_ref[k], u_aero[k])
    p_next = ca.vertcat(theta[k_next], kappa_ref[k_next], u_aero[k_next])
    p_c = 0.5 * (p_k + p_next)
    
    ds_k = ds_array[k] 
    
    f_k = f_physics(x_k, u_k, p_k)
    f_next = f_physics(x_next, u_next, p_next)
    
    x_c = 0.5 * (x_k + x_next) + (ds_k / 8.0) * (f_k - f_next)
    f_c = f_physics(x_c, u_c, p_c)
    
    x_integrated = x_k + (ds_k / 6.0) * (f_k + 4.0 * f_c + f_next)
    
    opti.subject_to(X_s[:, k+1] == x_integrated / scale_vector)
    
    #Path Bounds
    opti.subject_to(opti.bounded(10.0, v[k], 100.0))
    opti.subject_to(opti.bounded(-w_tr_right[k], n[k], w_tr_left[k]))
    opti.subject_to(opti.bounded(-0.6, xi[k], 0.6))    
    opti.subject_to(opti.bounded(-0.6, delta[k], 0.6)) 
    
    #Powertrain Limits
    gr_k = gear_discrete_fn(v[k])
    omega_engine_rads = (v[k] / R_wheel) * gr_k * Final_Drive
    rpm_k = omega_engine_rads * (30.0 / np.pi)
    
    x_rpm_norm = rpm_k / RPM_peak
    t_curve_Nm = T_peak * ((2.0 * x_rpm_norm) - x_rpm_norm**2)

    P_ICE_torque_limit = (t_curve_Nm * omega_engine_rads) / kW100_to_W 
    P_ICE_avail = smooth_min(P_ICE_torque_limit, P_ICE_max)
    
    opti.subject_to(opti.bounded(0.0, P_ICE[k], P_ICE_avail))

    v_kph = v[k] * 3.6
    if MODE == 'RACE':
        P_limit = (1800.0 - (5.0 * v_kph)) / 100.0 #Speed-Dependent Ramp Down Mandated by FIA
    else:
        P_limit = (7100.0 - (20.0 * v_kph)) / 100.0 
    
    #Constraints
    opti.subject_to(P_MGUK[k] <= P_limit)
    opti.subject_to(P_MGUK[k] <= P_MGUK_max_deploy)
    opti.subject_to(P_MGUK[k] >= -P_MGUK_max_regen)
    
    opti.subject_to(opti.bounded(0.0, F_brake_f[k], F_brake_max))
    opti.subject_to(opti.bounded(0.0, F_brake_r[k], F_brake_max))
    opti.subject_to(opti.bounded(SoC_min, soc[k], SoC_max))

    #Active Aero turns off when braking
    brake_viol_k = smooth_pos(((F_brake_f[k] + F_brake_r[k]) / F_brake_max) - 0.05)
    aero_shutoff_k = smooth_min(1.0, 100.0 * (brake_viol_k))
    u_aero_dyn_k = u_aero[k] * (1.0 - aero_shutoff_k)

    F_aero_k = 0.5 * rho * (ClA_Z - u_aero_dyn_k*Delta_ClA) * v[k]**2
    F_drag_k = 0.5 * rho * (CdA_Z - u_aero_dyn_k*Delta_CdA) * v[k]**2
    F_pt_k = (P_ICE[k]*kW100_to_W + 0.97 * P_MGUK[k]*kW100_to_W) / v[k]
    
    a_long_k = (F_pt_k - F_drag_k - (Crr*(m*g*ca.cos(theta[k]) + F_aero_k)) - (m*g*ca.sin(theta[k])) - F_brake_f[k] - F_brake_r[k]) / m
    
    W_transfer_k = (m * a_long_k + m * g * ca.sin(theta[k])) * (h_cog / L_base)

    F_z_f_raw = (m * g * ca.cos(theta[k]) * wd_f) + (F_aero_k * aero_bal_f) - W_transfer_k
    F_z_r_raw = (m * g * ca.cos(theta[k]) * (1.0 - wd_f)) + (F_aero_k * (1.0 - aero_bal_f)) + W_transfer_k
    
    F_z_f = smooth_floor(F_z_f_raw, 10.0)    
    F_z_r = smooth_floor(F_z_r_raw, 10.0)

    mu_f_long = p_dy1_long + (p_dy2 * ((F_z_f - Fz0_f) / Fz0_f))
    mu_r_long = p_dy1_long + (p_dy2 * ((F_z_r - Fz0_r) / Fz0_r))
    mu_f_lat = p_dy1_lat + (p_dy2 * ((F_z_f - Fz0_f) / Fz0_f))
    mu_r_lat = p_dy1_lat + (p_dy2 * ((F_z_r - Fz0_r) / Fz0_r))
    
    F_grip_max_f_long = mu_f_long * F_z_f
    F_grip_max_r_long = mu_r_long * F_z_r
    F_grip_max_f_lat = mu_f_lat * F_z_f
    F_grip_max_r_lat = mu_r_lat * F_z_r
    
    F_tire_long_f = -F_brake_f[k]                       
    F_tire_long_r = F_pt_k - F_brake_r[k]               
    
    kappa_car_eval = ca.tan(delta[k]) / L_base
    F_lat_req = m * (v[k]**2) * kappa_car_eval
    F_tire_lat_f = F_lat_req * (l_r / L_base)
    F_tire_lat_r = F_lat_req * (l_f / L_base)
    
    # Tyre friction circle
    opti.subject_to((F_tire_long_f / (F_grip_max_f_long + 1e-3))**2 + (F_tire_lat_f / (F_grip_max_f_lat + 1e-3))**2 <= 1.0 + sigma_tire[k])
    opti.subject_to((F_tire_long_r / (F_grip_max_r_long + 1e-3))**2 + (F_tire_lat_r / (F_grip_max_r_lat + 1e-3))**2 <= 1.0 + sigma_tire[k])
    
    if np.abs(kappa_ref[k]) > 1e-6:
        opti.subject_to(1.0 - (n[k] * kappa_ref[k]) >= 0.1) 
    
    #MGU-K Rate Limit
    if k < N - 1:
        dt_step = ds_k / v[k]
        k_next = (k + 1) % N
        k_prev = (k - 1) % N
        dP_MGUK = P_MGUK[k_next] - P_MGUK[k]
        #Ramp down bypass Conditions

        #Calculate whether car has braked
        brake_val_k = (F_brake_f[k] + F_brake_r[k]) / F_brake_max
        brake_val_k_next = (F_brake_f[k_next] + F_brake_r[k_next]) / F_brake_max
        brake_val_k_prev = (F_brake_f[k_prev] + F_brake_r[k_prev]) / F_brake_max
        
        sustained_brake = smooth_min(smooth_min(brake_val_k, brake_val_k_next), brake_val_k_prev)
        #Note: Ramp down can only be bypassed by braking for atleast three nodes, to prevent micro-braking by the optimizer
        
        #estimate throttle percentage using ICE Power to estimate lifting
        gr_next = gear_discrete_fn(v[k_next])
        omega_eng_next = (v[k_next] / R_wheel) * gr_next * Final_Drive
        rpm_next = omega_eng_next * (30.0 / np.pi)      
        x_rpm_next = rpm_next / RPM_peak
        t_curve_next = T_peak * ((2.0 * x_rpm_next) - x_rpm_next**2)
        P_ICE_avail_next = smooth_min((t_curve_next * omega_eng_next) / kW100_to_W, P_ICE_max)

        gr_prev = gear_discrete_fn(v[k_prev])
        omega_eng_prev = (v[k_prev] / R_wheel) * gr_prev * Final_Drive
        rpm_prev = omega_eng_prev * (30.0 / np.pi)
        x_rpm_prev = rpm_prev / RPM_peak
        t_curve_prev = T_peak * ((2.0 * x_rpm_prev) - x_rpm_prev**2)
        P_ICE_avail_prev = smooth_min((t_curve_prev * omega_eng_prev) / kW100_to_W, P_ICE_max)

        throttle_val_k = P_ICE[k] / (P_ICE_avail + 1e-3)
        throttle_val_k_next = P_ICE[k_next] / (P_ICE_avail_next + 1e-3)
        throttle_val_k_prev = P_ICE[k_prev] / (P_ICE_avail_prev + 1e-3)

        lift_now = 0.9 - throttle_val_k
        lift_next = 0.9 - throttle_val_k_next
        lift_prev = 0.9 - throttle_val_k_prev
        
        sustained_lift = smooth_min(smooth_min(lift_now, lift_next), lift_prev)
        #Ramp Down cannot be bypassed via lifting, even for three nodes, 
        #As optimizer will use micro lifts to bypass ramp-down into every braking zone
        #You can add this condition back if needed 


        # Standard instantaneous bypasses for physical limits
        bypass_brake = smooth_pos(sustained_brake - 0.1) #Ramp Down is lifted when braking
        bypass_speed = smooth_pos(58.33 - v[k]) / 58.33 #ramp Down applies only above 210kph
        bypass_power = smooth_pos(1.0 - P_MGUK[k]) #Applies only when MGUK is deploying more than 100kW
        bypass_regen = smooth_pos(-P_MGUK[k]) #Applies only when MGUK is deploying energy
        
        R_dynamic = R_deploy_ramp_down + 100.0 * (bypass_brake + bypass_speed + bypass_power + bypass_regen) 
        opti.subject_to(dP_MGUK >= -R_dynamic * dt_step)
        #Ramp-down is restricted to 100kW/s, unless the following exceptions in which case it spikes to well above 700kW/s, allowing bypass

        

opti.subject_to(t[0] == 0.0)
opti.subject_to(e_regen[0] == 0.0)

if MODE == 'QUALIFYING': #QUALI LAP, Spend all Energy, start full and end empty
    opti.subject_to(v[0] <= v_init)     
    opti.subject_to(soc[0] == SoC_max)
    opti.subject_to(soc[N] == SoC_min) 
    opti.subject_to(n[0] == n[N])
    opti.subject_to(xi[0] == xi[N])
    opti.subject_to(e_regen[N] <= E_regen_max_quali)

elif MODE == 'RACE': #RACE LAP, Balanced Deployment
    opti.subject_to(v[0] == v[N])     
    opti.subject_to(soc[0] == soc[N])
    opti.subject_to(n[0] == n[N])
    opti.subject_to(xi[0] == xi[N])
    opti.subject_to(e_regen[N] <= E_regen_max_race)

elif MODE == 'RACE_OT':
    opti.subject_to(v[0] == v[N])     
    opti.subject_to(soc[0] == soc[N])
    opti.subject_to(n[0] == n[N])
    opti.subject_to(xi[0] == xi[N])
    opti.subject_to(e_regen[N] <= E_regen_max_OT)
    


# 6. OBJECTIVE FUNCTION & REGULARIZATION

reg_weight = 1e-4
slack_weight = 50.0   # Massive penalty for cheating tire grip with slack variable

# 1. Steering smoothness
P_diff_penalty = ca.sumsqr(ca.diff(delta))

# 2. Tire slack penalty
slack_penalty = ca.sum1(sigma_tire)

# Minimize Time + Control Smoothness + Slack Violations
opti.minimize(t[N] + (reg_weight * P_diff_penalty)  + (slack_weight * slack_penalty))

# 7a. QSS INITIAL GUESS

warm_start_file = f'Warm Start/warm_start_{track}.npz'
use_warm_start = os.path.exists(warm_start_file)

if use_warm_start:
    try:
        ws = np.load(warm_start_file)
        X_prev_s, U_prev_s = ws['X_s'], ws['U_s']
        if X_prev_s.shape == (6, N + 1) and U_prev_s.shape == (5, N):
            print("Found warm_start.npz -- seeding from previous solution...")
            opti.set_initial(X_s, X_prev_s)
            opti.set_initial(U_s, U_prev_s)
            opti.set_initial(opti.lam_g, ws['lam_g'])
        else:
            print("warm_start.npz shape doesn't match current N -- falling back to QSS guess.")
            use_warm_start = False
    except KeyError:
        print("warm_start.npz contains old unscaled formatting -- falling back to QSS guess.")
        use_warm_start = False

if not use_warm_start:
    v_corner = np.full(N, 100.0)
    for i in range(N):
        if np.abs(kappa_ref[i]) > 1e-5:
            v_corner[i] = np.clip(np.sqrt((1.5 * g) / np.abs(kappa_ref[i])), 12.0, 100.0)

    v_fwd = np.copy(v_corner)
    a_accel_guess = 4.0  
    for i in range(N - 1):
        v_lim = np.sqrt(v_fwd[i]**2 + 2 * a_accel_guess * ds_array[i])
        v_fwd[i+1] = min(v_fwd[i+1], v_lim)

    v_bwd = np.copy(v_fwd)
    a_brake_guess = 5.0  
    for i in range(N - 1, 0, -1):
        v_lim = np.sqrt(v_bwd[i]**2 + 2 * a_brake_guess * ds_array[i-1])
        v_bwd[i-1] = min(v_bwd[i-1], v_lim)

    v_guess = np.append(v_bwd, v_bwd[0])   
    delta_guess = np.arctan(L_base * kappa_ref)

    dt_guess = ds_array / v_bwd
    t_guess = np.concatenate(([0.0], np.cumsum(dt_guess)))

    a_long_guess = np.diff(v_guess**2) / (2.0 * ds_array)   
    F_req_guess = m * a_long_guess   

    P_ICE_guess = np.zeros(N)
    P_MGUK_guess = np.zeros(N)
    F_brake_f_guess = np.zeros(N)
    F_brake_r_guess = np.zeros(N)

    for i in range(N):
        if F_req_guess[i] >= 0:
            P_req_kW100 = (F_req_guess[i] * v_bwd[i]) / kW100_to_W
            P_ICE_guess[i] = np.clip(P_req_kW100 * 0.6, 0.0, P_ICE_max)
            P_MGUK_guess[i] = np.clip(P_req_kW100 - P_ICE_guess[i], 0.0, P_MGUK_max_deploy)
        else:
            F_brake_total = -F_req_guess[i]
            F_brake_f_guess[i] = 0.45 * F_brake_total
            F_brake_r_guess[i] = 0.55 * F_brake_total

    P_MGUK_MW_guess = (P_MGUK_guess * kW100_to_W) / MW_to_W
    dsoc_guess = np.where(P_MGUK_MW_guess >= 0,
                       -P_MGUK_MW_guess,
                       0.92 * -P_MGUK_MW_guess) * dt_guess
    soc_guess = SoC_max - np.cumsum(np.concatenate(([0.0], dsoc_guess)))
    soc_guess = SoC_max + (soc_guess - soc_guess[0]) * ((SoC_min - SoC_max) / (soc_guess[-1] - soc_guess[0] + 1e-9))
    soc_guess = np.clip(soc_guess, SoC_min, SoC_max)

    # MAP GUESSES TO SCALED OPTIMIZER VARIABLES
    X_guess = np.vstack((t_guess, np.zeros(N+1), np.zeros(N+1), v_guess, soc_guess, np.zeros(N+1)))
    scale_x = np.array([TIME_SCALE, N_SCALE, XI_SCALE, V_SCALE, SOC_SCALE, E_REGEN_SCALE]).reshape(6,1)
    opti.set_initial(X_s, X_guess / scale_x)

    U_guess = np.vstack((P_ICE_guess, P_MGUK_guess, F_brake_f_guess, F_brake_r_guess, delta_guess))
    scale_u = np.array([P_ICE_SCALE, P_MGUK_SCALE, F_BRAKE_SCALE, F_BRAKE_SCALE, DELTA_SCALE]).reshape(5,1)
    opti.set_initial(U_s, U_guess / scale_u)

p_opts = {"expand": True}
s_opts = {
    "max_iter": 10000, 
    "print_level": 5,
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "acceptable_iter": 5,
    "linear_solver": "ma57",  #IF YOU HAVE ACCESS, USING CoinHSL. IF NOT, REMOVE
    "hessian_approximation": "limited-memory",
}
if use_warm_start:
    s_opts.update({
        "warm_start_init_point": "yes",
        "warm_start_bound_push": 1e-9,
        "warm_start_bound_frac": 1e-9,
        "warm_start_slack_bound_push": 1e-9,
        "warm_start_slack_bound_frac": 1e-9,
        "warm_start_mult_bound_push": 1e-9,
        "mu_init": 1e-5,
    })
opti.solver("ipopt", p_opts, s_opts)

# 8. SOLVE & POST-PROCESSING

try:
    print("Executing solver...")
    sol = opti.solve()
    print("\n--- 2026 F1 SIMULATION SUCCESSFUL ---")
    print(f"Optimal Lap Time:       {sol.value(t[N]):.3f} seconds")

    # --- VERIFY SLACK VARIABLE ---
    total_slack = np.sum(sol.value(sigma_tire))
    print(f"Total Slack Violation:  {total_slack:.6e}")
    
    if total_slack > 1e-4:
        print("WARNING: Solver used tire slack to converge. The lap time is physically invalid.")
    else:
        print("SUCCESS: Slack penalty is effectively zero. Trajectory is physically valid.")


    np.savez(f'Warm Start/warm_start_{track}.npz',
         X_s=sol.value(X_s),
         U_s=sol.value(U_s),
         lam_g=sol.value(opti.lam_g))
    print(f"Saved scaled solution to warm_start_{track}.npz for next run.")
    
    v_opt = sol.value(v) * 3.6
    n_opt = sol.value(n)
    
    e_regen_opt = sol.value(e_regen)
    soc_opt = sol.value(soc) 
    
    p_ice_opt = sol.value(P_ICE) * 100.0  
    p_mguk_opt = sol.value(P_MGUK) * 100.0
    p_deploy_opt = np.maximum(p_mguk_opt, 0.0)
    p_regen_opt = np.maximum(-p_mguk_opt, 0.0)
    
    v_plot = sol.value(v)[:-1]
    v_plot_kmh = v_plot * 3.6 
    np.savez(f'Results/sim_telemetry_{track}.npz', s=s_track, v_kmh=v_opt)

    f_brake_f_plot = sol.value(F_brake_f) / 1000.0
    f_brake_r_plot = sol.value(F_brake_r) / 1000.0

    # --- DYNAMIC AERO POST-PROCESSING ---
    u_aero_dyn_opt = np.zeros(N)
    p_ice_avail_opt = np.zeros(N)
    for i in range(N):
        brake_kn = (f_brake_f_plot[i] * 1000.0) + (f_brake_r_plot[i] * 1000.0)
        brake_viol = max(0.0, (brake_kn / F_brake_max) - 0.05)
        
        v_mps = v_plot[i]
        gr_i = float(gear_discrete_fn(v_mps))
        omega_eng = (v_mps / R_wheel) * gr_i * Final_Drive
        rpm = omega_eng * (30.0 / np.pi)
        x_rpm = rpm / RPM_peak
        t_curve = T_peak * ((2.0 * x_rpm) - x_rpm**2)
        p_ice_avail_kw = min((t_curve * omega_eng) / kW100_to_W, P_ICE_max) * 100.0
        p_ice_avail_opt[i] = p_ice_avail_kw
        
        
        aero_shutoff = min(1.0, 100.0 * brake_viol)
        u_aero_dyn_opt[i] = u_aero[i] * (1.0 - aero_shutoff)

    # --- REGEN BREAKDOWN CALCULATIONS ---
    t_opt_array = sol.value(t)
    dt_opt = np.diff(t_opt_array)
    
    energy_regen_step_MJ = 0.92 * (p_regen_opt / 1000.0) * dt_opt
    
    BRAKE_KN_THRESH = 3.0
    MGUK_KW_THRESH  = 5.0
    
    is_braking = (f_brake_f_plot + f_brake_r_plot) > BRAKE_KN_THRESH
    is_regen   = p_regen_opt > MGUK_KW_THRESH
    is_ice_full = (p_ice_opt >= (p_ice_avail_opt - 15.0)) & (p_ice_opt >= 380.0)
    is_ice_partial = ~is_ice_full
    
    e_brake_regen_MJ = np.sum(energy_regen_step_MJ[is_braking & is_regen])
    e_superclip_MJ   = np.sum(energy_regen_step_MJ[is_ice_full & is_regen & ~is_braking])
    e_part_regen_MJ  = (e_regen_opt[-1] - e_brake_regen_MJ - e_superclip_MJ)
    
    print("\n--- ENERGY HARVEST BREAKDOWN ---")
    print(f"Total Net Regen:        {e_regen_opt[-1]:.2f} MJ")
    print(f"  - Brake Regen:        {e_brake_regen_MJ:.2f} MJ")
    print(f"  - Superclipping:      {e_superclip_MJ:.2f} MJ")
    print(f"  - Partial Throttle:   {e_part_regen_MJ:.2f} MJ\n")



    # 9. DASHBOARD PLOTTING (INTERACTIVE PLOTLY)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Generating interactive Plotly dashboard...")
    
    s_plot_states = np.append(s_track, L_track)
    s_plot_controls = s_track
    lap_time_str = f"Lap Time: {sol.value(t[N]):.3f} s"

    # Create the vertically stacked subplots
    fig_dash = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=('Vehicle Speed', 'MGU-K Deploy & Regen', 'ICE Power', 'Battery SoC', 'Net Regen Harvested', 'Front Brake Force')
    )

    # 1. Speed Plot
    fig_dash.add_trace(go.Scatter(x=s_plot_states, y=v_opt, mode='lines', name='Speed [km/h]', line=dict(color=COL_PRIMARY, width=2)), row=1, col=1)

    # ADD STATIC BACKGROUND ZONES
    for i in range(len(s_plot_controls) - 1):
        if u_aero_dyn_opt[i] > 0.5:
            fig_dash.add_vrect(
                x0=s_plot_controls[i], 
                x1=s_plot_controls[i+1], 
                fillcolor=COL_AERO, 
                opacity=0.05,  # Very faint
                layer="below", 
                line_width=0, 
                row=1, col=1
            )

    # 2. MGU-K Deploy & Regen
    fig_dash.add_trace(go.Scatter(x=s_plot_controls, y=p_deploy_opt, fill='tozeroy', mode='lines', name='Deploy [kW]', line=dict(color=COL_DEPLOY)), row=2, col=1)
    fig_dash.add_trace(go.Scatter(x=s_plot_controls, y=-p_regen_opt, fill='tozeroy', mode='lines', name='Regen [kW]', line=dict(color=COL_REGEN)), row=2, col=1)

    # 3. ICE Power
    fig_dash.add_trace(go.Scatter(x=s_plot_controls, y=p_ice_opt, mode='lines', name='ICE Power [kW]', line=dict(color=COL_ICE, width=2)), row=3, col=1)
    fig_dash.add_trace(go.Scatter(x=s_plot_controls, y=p_ice_avail_opt, mode='lines', name='Avail ICE Power [kW]', line=dict(color=COL_ICE, width=2, dash='dot')), row=3, col=1)

    # 4. SoC
    fig_dash.add_trace(go.Scatter(x=s_plot_states, y=soc_opt, mode='lines', name='SoC [MJ]', line=dict(color=COL_SOC, width=2)), row=4, col=1)

    # 5. Net Regen
    fig_dash.add_trace(go.Scatter(x=s_plot_states, y=e_regen_opt, mode='lines', name='Net Regen [MJ]', line=dict(color=COL_NEUTRAL, width=2)), row=5, col=1)

    # 6. Front Brake Force
    fig_dash.add_trace(go.Scatter(x=s_plot_controls, y=f_brake_f_plot, mode='lines', name='Front Brake [kN]', line=dict(color=COL_BRAKE_F, width=2)), row=6, col=1)

    # Apply the dark theme to match the animation
    fig_dash.update_layout(
        title=f"2026 Telemetry Overview   —   {lap_time_str}",
        template="plotly_dark",
        height=1300,
        hovermode="x unified",
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=BG,
        font=dict(color=TEXT_C)
    )
    
    # Save to an HTML file and automatically open it in the default web browser
    dash_file = f"Dashboards/dashboard_{track}.html"
    fig_dash.write_html(dash_file, auto_open=True)
    print(f"Interactive dashboard opened in your web browser ({dash_file}).")

except RuntimeError:
    print("Solver failed to converge. Check the console log for Infeasibility (inf_pr) errors.")
 

if 'sol' in locals():
    print("Generating Live Track Animation...")
    import matplotlib.animation as animation
    from matplotlib.collections import LineCollection

    BRAKE_KN_THRESH = 3.0                  
    MGUK_KW_THRESH  = 10.0                 
    def classify_state(idx, prev_label):
        brake_kn = f_brake_f_plot[idx] 
        is_braking = brake_kn > BRAKE_KN_THRESH
        is_regen   = p_regen_opt[idx] > MGUK_KW_THRESH
        is_deploy  = p_deploy_opt[idx] > 30
        is_ice_full = (p_ice_opt[idx] >= (p_ice_avail_opt[idx] - 15.0)) and (p_ice_opt[idx] >= 380.0)
        is_fast = v_opt[idx] > 210

        if is_braking and is_regen:
            return 'Brake Regen', COL_REGEN_BRAKE
        if is_regen and not is_braking:
            if prev_label == 'Partial Regen':
                return 'Partial Regen', COL_PART_REGEN
            if prev_label == 'Superclip':
                return 'Superclip', COL_SUPERCLIP
            if is_ice_full and is_fast:
                return 'Superclip', COL_SUPERCLIP
            return 'Partial Regen', COL_PART_REGEN
        if is_deploy:
            return 'Deploy', COL_DEPLOY
        return 'Neutral', COL_NEUTRAL

    MIN_ZONE_LEN = 10.0  

    raw_labels = []
    raw_colors = []
    prev_label = 'Neutral'
    for i in range(N):
        lbl, col = classify_state(i, prev_label)
        raw_labels.append(lbl)
        raw_colors.append(col)
        prev_label = lbl
    runs = []  
    start = 0
    for i in range(1, N + 1):
        if i == N or raw_labels[i] != raw_labels[start]:
            length_m = ds_array[start:i].sum()
            runs.append([raw_labels[start], raw_colors[start], start, i, length_m])
            start = i

    filtered_runs = []
    for label, color, s_idx, e_idx, length_m in runs:
        if length_m < MIN_ZONE_LEN and filtered_runs:
            filtered_runs[-1][3] = e_idx          
            filtered_runs[-1][4] += length_m       
        else:
            filtered_runs.append([label, color, s_idx, e_idx, length_m])

    state_labels = [None] * N
    state_colors = [None] * N
    for label, color, s_idx, e_idx, length_m in filtered_runs:
        for i in range(s_idx, e_idx):
            state_labels[i] = label
            state_colors[i] = color
 
    dpsi = kappa_ref * ds_array
    psi = np.cumsum(dpsi)
    
    dx_c = np.cos(psi) * ds_array
    dy_c = np.sin(psi) * ds_array
    x_center = np.cumsum(dx_c)
    y_center = np.cumsum(dy_c)
    
    drift_x = np.linspace(0, x_center[-1] - x_center[0], len(x_center))
    drift_y = np.linspace(0, y_center[-1] - y_center[0], len(y_center))
    x_center = x_center - drift_x
    y_center = y_center - drift_y
    
    x_left = x_center - w_tr_left * np.sin(psi)
    y_left = y_center + w_tr_left * np.cos(psi)
    x_right = x_center + w_tr_right * np.sin(psi)
    y_right = y_center - w_tr_right * np.cos(psi)
    
    n_plot = n_opt[:-1]
    x_line = x_center - n_plot * np.sin(psi)
    y_line = y_center + n_plot * np.cos(psi)
 
    fig_anim = plt.figure(figsize=(12.0, 12.5), facecolor=BG)
    gs = fig_anim.add_gridspec(3, 2, width_ratios=[3.0, 1.0], height_ratios=[2.6, 1, 1], hspace=0.32, wspace=0.05,
                                left=0.04, right=0.98, top=0.94, bottom=0.06)
 
    ax_map = fig_anim.add_subplot(gs[0, 0])
    ax_speed = fig_anim.add_subplot(gs[1, 0])
    ax_pu = fig_anim.add_subplot(gs[2, 0], sharex=ax_speed)
    ax_side = fig_anim.add_subplot(gs[:, 1])
 
    for ax in (ax_map, ax_speed, ax_pu):
        ax.set_facecolor(PANEL_BG)
    ax_side.set_facecolor(BG)
    ax_side.axis('off')
 
    fig_anim.suptitle(f'{track}: 2026 F1 Lap Simulation', fontsize=17, fontweight='bold', color=TEXT_C)
 
    ribbon_x = np.concatenate([x_left, x_right[::-1]])
    ribbon_y = np.concatenate([y_left, y_right[::-1]])
    ax_map.fill(ribbon_x, ribbon_y, color=TARMAC, zorder=1)
    ax_map.plot(x_left, y_left, color=EDGE_C, linewidth=1.0, zorder=2)
    ax_map.plot(x_right, y_right, color=EDGE_C, linewidth=1.0, zorder=2)

    extend_m = 2.0 * (w_tr_left[0] + w_tr_right[0])   
    dx, dy = x_right[0] - x_left[0], y_right[0] - y_left[0]
    edge_len = np.hypot(dx, dy)
    ux, uy = dx / edge_len, dy / edge_len   

    sf_x0, sf_y0 = x_left[0]  - ux * extend_m, y_left[0]  - uy * extend_m
    sf_x1, sf_y1 = x_right[0] + ux * extend_m, y_right[0] + uy * extend_m

    ax_map.plot([sf_x0, sf_x1], [sf_y0, sf_y1], color='#000000', linewidth=7.5, solid_capstyle='butt', zorder=6.5, alpha=0.55)
    ax_map.plot([sf_x0, sf_x1], [sf_y0, sf_y1], color='#ffffff', linewidth=5.0, solid_capstyle='butt', zorder=7)
 
    aero_gap = max(14.0, 1.6 * np.mean(w_tr_right + w_tr_left))   
    offset_aero = w_tr_right + aero_gap
    x_aero = x_center + offset_aero * np.sin(psi)
    y_aero = y_center - offset_aero * np.cos(psi)
    aero_points = np.array([x_aero, y_aero]).T.reshape(-1, 1, 2)
    aero_segments = np.concatenate([aero_points[:-1], aero_points[1:]], axis=1)
    aero_active_idx = np.where(u_aero[:-1] != 0)[0]

    aero_handle = None
    if aero_active_idx.size > 0:
        for i in aero_active_idx:
            ax_map.plot([x_right[i], x_aero[i]], [y_right[i], y_aero[i]], color=COL_AERO, linewidth=0.5, alpha=0.3, linestyle=':', zorder=3)
        lc_aero = LineCollection(aero_segments[aero_active_idx], colors=COL_AERO, linewidths=1.6, linestyles=':', zorder=6)
        ax_map.add_collection(lc_aero)
        aero_handle = plt.Line2D([], [], color=COL_AERO, linewidth=1.6, linestyle=':')
 
    points = np.array([x_line, y_line]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_colors = state_colors[:N-1]   
 
    # Static colored line collection - never redrawn during animation
    lc = LineCollection(segments, colors=line_colors, linewidths=2.8, zorder=4)
    ax_map.add_collection(lc)
 
    ax_map.set_aspect('equal', adjustable='datalim')
    ax_map.margins(0.06)
    ax_map.axis('off')
    ax_map.autoscale_view()
 
    car_dot, = ax_map.plot([], [], 'o', color=COL_CAR, markersize=12, markeredgecolor='black', markeredgewidth=1.2, zorder=6)
 
    legend_handles = [plt.Line2D([], [], color=c, linewidth=2.8) for c in [COL_DEPLOY, COL_REGEN_BRAKE, COL_PART_REGEN, COL_SUPERCLIP, COL_NEUTRAL]]
    legend_labels = ['Deploy', 'Brake Regen', 'Partial Throttle Regen', 'Superclip', 'Clipping/No Deployment']
    if aero_handle is not None:
        legend_handles.append(aero_handle)
        legend_labels.append('Active Aero')
 
    ax_side.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False, labelcolor=TEXT_C, fontsize=10, borderaxespad=0.)
 
    hud_text = ax_side.text(0.5, 0.77, '', transform=ax_side.transAxes, horizontalalignment='center', verticalalignment='top', fontsize=9.5, fontfamily='monospace', color=TEXT_C,
                           bbox=dict(facecolor='#1a1d24', alpha=0.9, edgecolor=EDGE_C, boxstyle='round,pad=0.6'))

    summary_str = (
        f"Lap Time: {sol.value(t[N]):.3f}s\n\n"
        f"Regen Breakdown:\n"
        f"  Total:   {e_regen_opt[-1]:.3f} MJ\n"
        f"  Brake:   {e_brake_regen_MJ:.3f} MJ\n"
        f"  Superclip: {e_superclip_MJ:.3f} MJ\n"
        f"  Partial: {e_part_regen_MJ:.3f} MJ"
    )
    ax_side.text(0.5, 0.40, summary_str, transform=ax_side.transAxes, horizontalalignment='center', verticalalignment='top', fontsize=10.5, fontfamily='monospace', color=TEXT_C,
                 bbox=dict(facecolor='#1a1d24', alpha=0.9, edgecolor=COL_REGEN, boxstyle='round,pad=0.6'))
 
    speed_points = np.array([s_plot_controls, v_plot_kmh]).T.reshape(-1, 1, 2)
    speed_segments = np.concatenate([speed_points[:-1], speed_points[1:]], axis=1)
    lc_speed = LineCollection(speed_segments, colors=line_colors, linewidths=2.5, zorder=4)
    ax_speed.add_collection(lc_speed)
    ax_speed.set_xlim(0, L_track)
    ax_speed.set_ylim(0, np.max(v_plot_kmh) * 1.12)
    style_axis(ax_speed, ylabel='Speed [km/h]')
    ax_speed.tick_params(labelbottom=False)
 
    vline_speed, = ax_speed.plot([], [], color=COL_CAR, linewidth=1.6, alpha=0.9, zorder=3)
    car_speed_dot, = ax_speed.plot([], [], 'o', color=COL_CAR, markersize=7, markeredgecolor='black', zorder=5)
 
    ax_pu.fill_between(s_plot_controls, 0, p_deploy_opt, color=COL_DEPLOY, alpha=0.45, label='Deploy [kW]')
    ax_pu.fill_between(s_plot_controls, 0, p_regen_opt, color=COL_REGEN, alpha=0.45, label='Regen [kW]')
    ax_pu.set_xlim(0, L_track)
    ax_pu.set_ylim(0, P_MGUK_max_regen * 100.0 * 1.12)
    style_axis(ax_pu, xlabel='Distance around lap [m]', ylabel='MGU-K Power [kW]')
 
    ax_soc = ax_pu.twinx()
    ax_soc.plot(s_plot_states, soc_opt, color=COL_SOC, linewidth=2)
    ax_soc.set_ylim(0, SoC_max + 0.5)
    ax_soc.set_facecolor(PANEL_BG)
    ax_soc.set_ylabel('SoC [MJ]', color=COL_SOC)
    ax_soc.tick_params(axis='y', colors=COL_SOC)
    ax_soc.spines['right'].set_color(COL_SOC)
 

 
    vline_pu, = ax_pu.plot([], [], color=COL_CAR, linewidth=1.6, alpha=0.9, zorder=3)
    car_pu_dot, = ax_pu.plot([], [], 'o', color=COL_CAR, markersize=7, markeredgecolor='black', zorder=5)


    hud_frames = []
    p_max_frames = np.maximum(p_deploy_opt, p_regen_opt)
    for idx in range(N):
        speed = v_plot_kmh[idx]
        ice_pwr = p_ice_opt[idx]
        mguk_pwr = p_mguk_opt[idx]
        current_soc = soc_opt[idx]
        mode = f"{state_labels[idx].upper()}"
        aero_state = "OPEN" if u_aero[idx] != 0 else "CLOSED"
        
        hud_frames.append(
            f"SPEED   {speed:4.0f} km/h\n"
            f"ICE     {ice_pwr:4.0f} kW\n"
            f"MGUK    {mguk_pwr:4.0f} kW\n"
            f"MODE    {mode}\n"
            f"AERO    {aero_state}\n"
            f"SoC     {current_soc:4.2f} MJ"
        )
 
    def init():
        car_dot.set_data([], [])
        car_speed_dot.set_data([], [])
        car_pu_dot.set_data([], [])
        vline_speed.set_data([], [])
        vline_pu.set_data([], [])
        hud_text.set_text('')
        return car_dot, car_speed_dot, car_pu_dot, vline_speed, vline_pu, hud_text
 
    def update(frame):
        idx = frame % N
        car_dot.set_data([x_line[idx]], [y_line[idx]])
 
        s_now = s_plot_controls[idx]
        car_speed_dot.set_data([s_now], [v_plot_kmh[idx]])
        car_pu_dot.set_data([s_now], [p_max_frames[idx]])
        
        vline_speed.set_data([s_now, s_now], ax_speed.get_ylim())
        vline_pu.set_data([s_now, s_now], ax_pu.get_ylim())
        
        hud_text.set_text(hud_frames[idx])
 
        return car_dot, car_speed_dot, car_pu_dot, vline_speed, vline_pu, hud_text
 

    ani = animation.FuncAnimation(fig_anim, update, frames=N,
                                  init_func=init, blit=True, interval=16, repeat=True)
 
    plt.show()
