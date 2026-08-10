# F1-Laptime-Optimizer

This repository is a lap time sim tool for F1 2026 cars. Uses CasADi + IPOPT. Models the 2026 Regs 

## Features

* Hermite-Simpson dynamics loop

* Ingests raw track coordinates from CSV files and fits periodic B-Splines to smooth the centerline. It applies adaptive mesh refinement, calculating track curvature to place denser nodes in tight corners and sparser nodes on straights.

* `fastf1` library is used to get track elevation (Z) data, syncing it to the custom track coordinates using KDTree spatial matching.

* Maps active aero zones from CSV files directly onto the track map. 

---

## Prerequisites & Installation


**Dependencies:**

* `casadi`

* `numpy`

* `matplotlib`

* `scipy`

* `pandas`

* `fastf1`

* `plotly`


> **NOTE:** By default, the CasADi Ipopt solver is configured to use the **MA57 linear solver**, which requires an academic license (CoinHSL). If you do not have this installed on your machine, you must remove `"linear_solver": "ma57"` from the solver options before running the main simulation.
> 
> 

---

## Usage

The project is designed to be run in a three-step pipeline:

### 1. Track Processing

Run the track processor script to generate track mesh.

* **Inputs:** Requires raw track coordinates (`Circuit Data/Raw Coordinates/{TRACK}.csv`) and active aero zones (`Circuit Data/Active Aero Zones/{TRACK}_aero_zones.csv`).


* **Output:** Generates a compressed `.npz` file 


### 2. Lap Simulation

Run the main CasADi optimizer script to calculate the optimal lap time and energy deployment.

* **Configuration:** At the top of the file, you can set the `track` name, Max Regen per lap, starting speed for a Quali Lap. the PU regulation `YEAR` (2026, 2027, or 2028), and the optimization `MODE` (QUALIFYING, RACE, RACE_OT).


* **Warm Start:** The script will automatically look for a `warm_start_{track}.npz` file. If none exists, it calculates a quasi-steady-state (QSS) initial guess.


* **Outputs:** Saves the simulated telemetry to `Results/sim_telemetry_{track}.npz`, opens a Plotly HTML dashboard, and renders a live telemetry animation.

### 3. Telemetry Correlation

Run the comparison script to validate the sim vs real Formula 1 data.

* **Configuration:** Define the `YEAR`, `TRACK`, and `SESSION` (e.g., 'Q' for Qualifying) to fetch the absolute fastest lap using `fastf1`.


* **Output:** Generates a two-pane matplotlib figure comparing the CasADi simulation against the real telemetry, including an auto-aligned speed delta plot.
