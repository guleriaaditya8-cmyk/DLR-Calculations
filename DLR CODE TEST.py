import pandapower as pp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandapower.create import (
    create_empty_network,
    create_bus,
    create_ext_grid,
    create_load,
    create_line,
    create_transformer,
    create_gen,
    create_sgen
)
from pandapower.run import runpp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
from weather_api import load_open_meteo_weather, prompt_for_weather_date



# ==============================
# DLR FORMULA PARAMETERS
# ==============================
# Pj = I^2 R, Ps = alpha_s S D, Pc = E V^0.6 (Tc - Ta),
# Pr = 17.8 D epsilon (Tc - Ta), and Pj + Ps = Pc + Pr.
D_mm = 15.65
D_m = D_mm / 1000
epsilon = 0.6
alpha_s = 0.5
E = 3.645
T_conductor_c = 80
R_20 = 0.054 / 1000
alpha = 0.00403
STATIC_AMPACITY = 626
MAX_DLR_AMPACITY = 2200

# Goldwind GW 136-4.8MW datasheet values.
WIND_TURBINE_RATED_POWER_MW = 4.8
WIND_TURBINE_COUNT = 100
WIND_CUT_IN_MS = 2.5
WIND_RATED_MS = 12.5
WIND_CUT_OUT_MS = 26.0
WIND_RATED_POWER_MW = WIND_TURBINE_RATED_POWER_MW * WIND_TURBINE_COUNT
WIND_HUB_HEIGHT_M = 110.0

# Huawei SUN2000-100KTL-M2 datasheet values.
PV_INVERTER_NOMINAL_AC_KW = 100.0
PV_INVERTER_MAX_AC_KW = 110.0
PV_INVERTER_COUNT = 2000
PV_INVERTER_EFFICIENCY = 0.986
PV_RATED_POWER_MW = (
    PV_INVERTER_NOMINAL_AC_KW * PV_INVERTER_COUNT / 1000.0
)
PV_MAX_OUTPUT_MW = PV_INVERTER_MAX_AC_KW * PV_INVERTER_COUNT / 1000.0
PV_DC_ARRAY_RATED_MW = PV_RATED_POWER_MW / PV_INVERTER_EFFICIENCY

WEATHER_DATE = prompt_for_weather_date()


def load_local_weather(selected_date):
    """Load the same 24-hour, 100 m weather profile from Open-Meteo."""
    weather = load_open_meteo_weather(selected_date).rename(columns={
        "datetime": "time",
        "wind_speed_100m": "wind_speed_ms",
        "cloud_cover": "cloud_cover_percent",
    })
    return weather


print(f"\n[INFO] Loading local Paris weather data for {WEATHER_DATE}...")
df_weather = load_local_weather(WEATHER_DATE)
WEATHER_DATE = df_weather["time"].iloc[0].date().isoformat()
print(df_weather)
print("\n[INFO] Paris weather summary:")
print(df_weather[["temperature_c", "wind_speed_ms", "cloud_cover_percent"]].describe())

temperature_profile = df_weather["temperature_c"].tolist()
wind_profile = df_weather["wind_speed_ms"].tolist()
cloud_profile = df_weather["cloud_cover_percent"].tolist()
simulation_hours = 24


def calculate_dlr_ampacity(temperature_c, wind_speed, cloud_cover):
    delta_t = max(T_conductor_c - temperature_c, 0)
    wind = min(max(wind_speed, 0), 10)

    r_tc = R_20 * (1 + alpha * (T_conductor_c - 20))
    solar_radiation = 1000 * (1 - 0.75 * (cloud_cover / 100))
    ps = alpha_s * solar_radiation * D_m
    # The diameter factor keeps convective cooling expressed per conductor
    # length; without it the result is unrealistically forced to 2200 A.
    pc = E * (D_m ** 0.75) * (wind ** 0.6) * delta_t
    pr = 17.8 * D_m * epsilon * delta_t

    ampacity = np.sqrt(max((pc + pr - ps) / r_tc, 0))
    ampacity = max(ampacity, STATIC_AMPACITY)
    return min(ampacity, MAX_DLR_AMPACITY)


dlr_ampacity_profile = [
    calculate_dlr_ampacity(temperature_profile[hour],
                           wind_profile[hour],
                           cloud_profile[hour])
    for hour in range(simulation_hours)
]


# define power factor
pf = 0.95

phi = np.arccos(pf)   # corrected
q_factor = np.tan(phi)

print("Reactive Power factor is:", q_factor)

p_mw = 3
q_mvar = p_mw * q_factor

print("Reactive Power is:", q_mvar)


def run_power_flow(network):
    """Use the same Newton solver settings as Transmission_line.py."""
    init = "results" if network.get("converged", False) else "dc"
    pp.runpp(
        network,
        algorithm="nr",
        init=init,
        max_iteration=50,
        tolerance_mva=1e-6,
    )


def calculate_line_loading_percent(network, line, ampacity_a):
    """Calculate loading from the higher current at either line terminal."""
    terminal_current_ka = max(
        network.res_line.at[line, "i_from_ka"],
        network.res_line.at[line, "i_to_ka"],
    )
    return terminal_current_ka / (ampacity_a / 1000) * 100


# create empty net
net = pp.create_empty_network()

# create buses
b1 = pp.create_bus(net, vn_kv=15, name="Bus PV")
b2 = pp.create_bus(net, vn_kv=225, name="Bus PV Step Up")
b3 = pp.create_bus(net, vn_kv=225, name="Bus PV Step Down")
b4 = pp.create_bus(net, vn_kv=20, name="Bus PV Load")
b5 = pp.create_bus(net, vn_kv=225, name="Bus Feeder")
b6 = pp.create_bus(net, vn_kv=225, name="Bus Step Down to Feeder Load")
b7 = pp.create_bus(net, vn_kv=20, name="Bus feeder load distribution")
b8 = pp.create_bus(net, vn_kv=20, name="Bus Wind step up")
b9 = pp.create_bus(net, vn_kv=225, name="Bus wind Transmission")
b10 = pp.create_bus(net, vn_kv=225, name="Bus Wind Destribution")
b11 = pp.create_bus(net, vn_kv=20, name="Bus Feeder Destribution")

# capacitence for line RL is zero for transmission line and 0.1 for distribution line to simulate the effect of capacitance in distribution line and its effect on voltage profile and power flow of the system
pp.create_std_type(net, {
    "c_nf_per_km": 12,
    "r_ohm_per_km": 0.025,
    "x_ohm_per_km": 0.32,
    "max_i_ka": STATIC_AMPACITY / 1000
}, name="225kV_line", element="line")



# Transformer parameters used by the 225 kV PV/wind network.  The standard
# type names intentionally include the actual nameplate rating so exported and
# plotted results cannot be mistaken for the old 10/15 MVA labels.

pp.create_std_type(net, {
    "sn_mva": 400,
    "vn_hv_kv": 225,
    "vn_lv_kv": 20,
    "vk_percent": 10,
    "vkr_percent": 0.3,
    "pfe_kw": 5,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-2 400 MVA 225/20 kV", element="trafo")


pp.create_std_type(net, {
    "sn_mva": 505,
    "vn_hv_kv": 225,
    "vn_lv_kv": 20,
    "vk_percent": 10,
    "vkr_percent": 0.3,
    "pfe_kw": 5,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-4 505 MVA 225/20 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 210,
    "vn_hv_kv": 225,
    "vn_lv_kv": 15,
    "vk_percent": 12,
    "vkr_percent": 0.4,
    "pfe_kw": 8,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-1 210 MVA 225/15 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 231,
    "vn_hv_kv": 225,
    "vn_lv_kv": 20,
    "vk_percent": 10,
    "vkr_percent": 0.4,
    "pfe_kw": 8,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-3 231 MVA 225/20 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 336, "vn_hv_kv": 225, "vn_lv_kv": 20,
    "vk_percent": 10, "vkr_percent": 0.3, "pfe_kw": 5,
    "i0_percent": 0.1, "shift_degree": 0
}, name="TR-5 336 MVA 225/20 kV", element="trafo")



load_id1 = pp.create_load(net, bus=b4, p_mw=300, q_mvar=300 * q_factor, name="Zone 1 Load")
load_id2 = pp.create_load(net, bus=b7, p_mw=20, q_mvar=20 * q_factor, name="Zone 2 Load A")
load_id3 = pp.create_load(net, bus=b7, p_mw=80, q_mvar=80 * q_factor, name="Zone 2 Load B")
load_id4 = pp.create_load(net, bus=b7, p_mw=120, q_mvar=120 * q_factor, name="Zone 2 Load C")
load_id5 = pp.create_load(net, bus=b11, p_mw=320, q_mvar=320 * q_factor, name="Zone 3 Load")

 # create load for pv line to test the power flow and voltage profile of the system with 3 MW load and reactive power of 0.99 MVAR

#creating generator PV // Zone 1 with high-voltage smart-grid scale photovoltaic generation
# The replicated profiles peak at 200 MW PV and 384 MW wind.  The 200 MW and
# 480 MW nameplates retain reactive limits from the reference network.
PV_Gen = create_sgen(net, bus=b1, p_mw=200, max_q_mvar=65.7, min_q_mvar=-65.7, name="PV Generator 200 MW")

# create transformers to step up from PV to 132KV and step down from 132KV to 33KV for destribution
trfo_id1 = pp.create_transformer(net, hv_bus=b2, lv_bus=b1, std_type="TR-1 210 MVA 225/15 kV", name="TR-1 PV Step Up")

line_id1 = pp.create_line(net, from_bus=b2, to_bus=b3, length_km=35, name="Line 1 - PV transmission",std_type="225kV_line")

trfo_id2 = pp.create_transformer(net, hv_bus=b3, lv_bus=b4, std_type="TR-2 400 MVA 225/20 kV", name="TR-2 Zone 1 Step Down")
# creating load for pv line to test the power flow and voltage profile of the system with 3 MW load and reactive power of 0.99 MVAR


#inter bus connection between PV transmission line bus 132KV and feeder transmission line // Zone 1 and Zone 2
line_id2 = pp.create_line(net, from_bus=b2, to_bus=b6, length_km=60, name="Line 2 - PV to Zone 2",std_type="225kV_line")




#creating generator PV // Zone 2 with Feeder
create_ext_grid(net, bus=b2, vm_pu=1.02, va_degree=0, name="Main 225 kV grid interconnection")
create_ext_grid(net, bus=b5, vm_pu=1.02, va_degree=0, name="Feeder 225 kV grid interconnection")

# creating transmission line for feeder connction
line_id3 = pp.create_line(net, from_bus=b5, to_bus=b6, length_km=60, name="Line 3 - feeder transmission",std_type="225kV_line")
# creating transformers for step down from feeder to destribution
trfo_id3 = pp.create_transformer(net, hv_bus=b6, lv_bus=b7, std_type="TR-3 231 MVA 225/20 kV", name="TR-3 Zone 2 Step Down")

#interconnection between feeder transmission line bus 132KV and wind transmission line bus 132KV for Zone 2 and Zone 3
line_id4 = pp.create_line(net, from_bus=b6, to_bus=b9, length_km=60, name="Line 4 - feeder to wind",std_type="225kV_line")






#creating generator Wind // Zone 3 with high-voltage smart-grid scale wind generation
Wind_Gen = create_sgen(net, bus=b8, p_mw=480, max_q_mvar=154, min_q_mvar=-154, name="Wind Generator 480 MW")

# creating feeder bus connection to wind bus for step up transformer
 
trfo_id5 = pp.create_transformer(net, hv_bus=b9, lv_bus=b8, std_type="TR-4 505 MVA 225/20 kV", name="TR-4 Wind Step Up")

line_id5 = pp.create_line(net, from_bus=b9, to_bus=b10, length_km=60, name="Line 5 - wind transmission",std_type="225kV_line")

trfo_id6 = pp.create_transformer(net, hv_bus=b10, lv_bus=b11, std_type="TR-5 336 MVA 225/20 kV", name="TR-5 Zone 3 Step Down")




#define load profile for each load to simulate the effect of load variation on the power flow and voltage profile of the system-------------------------------------------

load_shape = [.45, .43, .42, .41, .41, .44, .50, .54, .57, .60, .61, .62,
              .63, .73, .74, .74, .72, .75, .70, .65, .63, .59, .53, .48]
nominal_load_mw = {load_id1: 300, load_id2: 20, load_id3: 80,
                   load_id4: 120, load_id5: 320}
load_profiles = {load_id: [nominal * factor for factor in load_shape]
                 for load_id, nominal in nominal_load_mw.items()}
LOAD_SCALE = 1


def calculate_wind_power_mw(wind_speed_ms):
    if wind_speed_ms < WIND_CUT_IN_MS or wind_speed_ms >= WIND_CUT_OUT_MS:
        return 0.0
    if wind_speed_ms >= WIND_RATED_MS:
        return WIND_RATED_POWER_MW

    normalized_speed = (
        (wind_speed_ms - WIND_CUT_IN_MS)
        / (WIND_RATED_MS - WIND_CUT_IN_MS)
    )
    return WIND_RATED_POWER_MW * normalized_speed ** 3

weather_hour = df_weather["time"].dt.hour
solar_angle = np.maximum(np.sin(np.pi * (weather_hour - 6) / 12), 0.0)
clear_sky_irradiance = 1000.0 * solar_angle
cloud_factor = 1.0 - 0.75 * df_weather["cloud_cover_percent"] / 100.0
temperature_factor = 1.0 - 0.004 * (df_weather["temperature_c"] - 25.0)
temperature_factor = temperature_factor.clip(lower=0.0)
solar_irradiance = clear_sky_irradiance * cloud_factor
pv_generation_profile = (
    PV_DC_ARRAY_RATED_MW
    * (solar_irradiance / 1000.0)
    * temperature_factor
    * PV_INVERTER_EFFICIENCY
).clip(lower=0.0, upper=PV_MAX_OUTPUT_MW).round(2).tolist()
wind_generation_profile = [
    calculate_wind_power_mw(wind_speed_ms)
    for wind_speed_ms in wind_profile
]

wind_generation_status = []
for wind_speed_ms, generation_mw in zip(wind_profile, wind_generation_profile):
    if wind_speed_ms < WIND_CUT_IN_MS:
        status = "below cut-in"
    elif wind_speed_ms >= WIND_CUT_OUT_MS:
        status = "above cut-out"
    elif wind_speed_ms >= WIND_RATED_MS:
        status = "rated output"
    else:
        status = "partial output"
    wind_generation_status.append(status)

print("\n--- Smart Grid Study Setup ---")
print("Load profile scale factor:", LOAD_SCALE, "(profiles are per-unit of nominal load)")
print("Weather-driven PV generation range (MW):", min(pv_generation_profile), "to", max(pv_generation_profile))
print("Wind generation range (MW):", min(wind_generation_profile), "to", max(wind_generation_profile))
print("Wind generation hours by operating state:")
for status in ("below cut-in", "partial output", "rated output", "above cut-out"):
    print(f"  {status}: {wind_generation_status.count(status)}")
print("PV inverter configuration:", PV_INVERTER_COUNT, "x", PV_INVERTER_NOMINAL_AC_KW, "kW")
print("PV nominal/max AC output (MW):", PV_RATED_POWER_MW, "/", PV_MAX_OUTPUT_MW)
print("Wind turbine configuration:", WIND_TURBINE_COUNT, "x", WIND_TURBINE_RATED_POWER_MW, "MW")
print("Wind hub-height input approximation (m):", WIND_HUB_HEIGHT_M, "using 120 m weather data")


time = range(simulation_hours)

lines = [line_id1, line_id2, line_id3, line_id4, line_id5]
net.line.loc[lines, "in_service"] = True

# Match the clean report-style appearance of the reference transmission study.
plt.rcParams.update({
    "figure.facecolor": "#f7f9fc", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#1e293b",
    "axes.titleweight": "bold", "axes.titlesize": 14, "axes.labelsize": 11,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "grid.color": "#dbe3ee", "grid.linestyle": "--", "grid.alpha": 0.7,
    "legend.frameon": True, "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cbd5e1", "font.size": 10,
})
PLOT_COLORS = plt.get_cmap("tab10").colors
PLOT_PALETTE = {
    "temperature": "#e76f51",
    "wind_speed": "#2a9d8f",
    "cloud_cover": "#7b2cbf",
    "static_ampacity": "#264653",
    "dlr_ampacity": "#457b9d",
    "slr_loading": "#d1495b",
    "dlr_loading": "#0077b6",
    "warning_limit": "#f4a261",
    "congestion_limit": "#6d6875",
    "loading_difference": "#2a9d8f",
    "zero_reference": "#adb5bd",
    "total_generation": "#2a9d8f",
    "total_load": "#264653",
    "pv_generation": "#e9c46a",
    "wind_generation": "#457b9d",
    "grid_exchange": "#7b2cbf",
    "system_losses": "#f9844a",
    "grid_import": "#277da1",
    "grid_export": "#43aa8b",
    "renewable_penetration": "#90be6d",
    "full_renewable": "#6d6875",
}


def finish_time_axis(axis, title, ylabel):
    """Apply the shared reference-chart labels, grid, and date formatting."""
    axis.set_title(title, pad=14)
    axis.set_xlabel("Date and Time")
    axis.set_ylabel(ylabel)
    axis.grid(True)
    axis.legend(loc="best")
    axis.tick_params(axis="x", rotation=45)

slr_line_loading_history = {line: [] for line in lines}
dlr_line_loading_history = {line: [] for line in lines}
line_loading_difference = {line: [] for line in lines}
hourly_results = []

for hour in range(simulation_hours):
    profile_hour = hour % 24

    # update loads
    for load_id in load_profiles:
        net.load.at[load_id, "p_mw"] = load_profiles[load_id][profile_hour]
        net.load.at[load_id, "q_mvar"] = (
            load_profiles[load_id][profile_hour] * q_factor
        )

    net.sgen.at[PV_Gen, "p_mw"] = pv_generation_profile[profile_hour]
    net.sgen.at[Wind_Gen, "p_mw"] = wind_generation_profile[profile_hour]

    # SLR run with fixed static ampacity
    net.line.loc[lines, "max_i_ka"] = STATIC_AMPACITY / 1000
    run_power_flow(net)
    for line in lines:
        slr_loading = calculate_line_loading_percent(net, line, STATIC_AMPACITY)
        pandapower_slr_loading = net.res_line.loading_percent[line]
        if not np.isclose(slr_loading, pandapower_slr_loading, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"SLR loading mismatch on line {line}: "
                f"calculated {slr_loading:.6f}%, "
                f"pandapower {pandapower_slr_loading:.6f}%"
            )
        slr_line_loading_history[line].append(slr_loading)

    # DLR run with weather-dependent ampacity
    net.line.loc[lines, "max_i_ka"] = dlr_ampacity_profile[hour] / 1000
    run_power_flow(net)
    for line in lines:
        dlr_loading = calculate_line_loading_percent(
            net,
            line,
            dlr_ampacity_profile[hour],
        )
        slr_loading = slr_line_loading_history[line][-1]
        dlr_line_loading_history[line].append(dlr_loading)
        line_loading_difference[line].append(slr_loading - dlr_loading)

    # Keep a complete record for report-style tables as well as graphs.
    hourly_record = {
        "time": df_weather.at[hour, "time"],
        "hour": hour,
        "profile_hour": profile_hour,
        "total_load_mw": net.load["p_mw"].sum(),
        "pv_mw": net.sgen.at[PV_Gen, "p_mw"],
        "wind_mw": net.sgen.at[Wind_Gen, "p_mw"],
        "grid_mw": net.res_ext_grid["p_mw"].sum(),
        "losses_mw": net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum(),
        "min_voltage_pu": net.res_bus["vm_pu"].min(),
            "weakest_bus": net.bus.at[net.res_bus["vm_pu"].idxmin(), "name"],
            "max_slr_loading_percent": max(slr_line_loading_history[line][-1] for line in lines),
            "max_dlr_loading_percent": max(dlr_line_loading_history[line][-1] for line in lines),
    }
    for line in lines:
        line_name = net.line.at[line, "name"]
        hourly_record[f"{line_name} SLR (%)"] = slr_line_loading_history[line][-1]
        hourly_record[f"{line_name} DLR (%)"] = dlr_line_loading_history[line][-1]
    for transformer in net.trafo.index:
        transformer_name = net.trafo.at[transformer, "name"]
        hourly_record[f"{transformer_name} loading (%)"] = net.res_trafo.at[
            transformer, "loading_percent"
        ]
    hourly_results.append(hourly_record)

df_study_results = pd.DataFrame(hourly_results)
print("Maximum Daily SLR Loading (%):", round(float(df_study_results["max_slr_loading_percent"].max()), 2))
print("Maximum Daily DLR Loading (%):", round(float(df_study_results["max_dlr_loading_percent"].max()), 2))
print("Minimum Daily Voltage (pu):", round(float(df_study_results["min_voltage_pu"].min()), 4))
weakest_hour = df_study_results["min_voltage_pu"].idxmin()
print("Weakest Bus During Study:", df_study_results.at[weakest_hour, "weakest_bus"])

line_loading_history = slr_line_loading_history
net.line.loc[lines, "max_i_ka"] = STATIC_AMPACITY / 1000

print("\n--- SLR vs DLR Loading Difference ---")
print("Static ampacity (A):", STATIC_AMPACITY)
print("Average DLR ampacity (A):", round(float(np.mean(dlr_ampacity_profile)), 2))
print("Maximum DLR ampacity (A):", round(float(np.max(dlr_ampacity_profile)), 2))
print(
    "Hours at maximum DLR limit:",
    sum(ampacity >= MAX_DLR_AMPACITY for ampacity in dlr_ampacity_profile),
    "/",
    simulation_hours,
)

for line in lines:
    avg_slr = np.mean(slr_line_loading_history[line])
    avg_dlr = np.mean(dlr_line_loading_history[line])
    avg_reduction = np.mean(line_loading_difference[line])
    print(
        f"{net.line.loc[line, 'name']} | "
        f"Avg SLR {avg_slr:.2f}% | "
        f"Avg DLR {avg_dlr:.2f}% | "
        f"Reduction {avg_reduction:.2f} percentage points"
    )





# run power flow to get the results for the initial load profile-----------------------------------------------------------------------------------------------



run_power_flow(net)

#simulating load flow of each transmission line
net.res_line

# Use the element ID returned when the line was created.  This is robust to
# display-name edits and avoids an empty pandas selection / IndexError.
line_index = line_id5
print("Line Name:", net.line.loc[line_index, "name"])

print("\n--- Voltage ---")
print("From Bus Voltage (pu):", net.res_line.loc[line_index, "vm_from_pu"])
print("To Bus Voltage (pu):", net.res_line.loc[line_index, "vm_to_pu"])

print("\n--- Current ---")
print("From Bus Current (kA):", net.res_line.loc[line_index, "i_from_ka"])
print("To Bus Current (kA):", net.res_line.loc[line_index, "i_to_ka"])

print("\n--- Power Flow ---")
print("P from (MW):", net.res_line.loc[line_index, "p_from_mw"])
print("Q from (MVAR):", net.res_line.loc[line_index, "q_from_mvar"])

print("\n--- Final SLR Loading ---")
print("Loading (%):", net.res_line.loc[line_index, "loading_percent"])



#Voltage in kv
vn = net.bus.loc[net.line.loc[line_index, "from_bus"], "vn_kv"]

v_from_kv = net.res_line.loc[line_index, "vm_from_pu"] * vn
v_to_kv = net.res_line.loc[line_index, "vm_to_pu"] * vn

print("From Voltage (kV):", v_from_kv)
print("To Voltage (kV):", v_to_kv)

#apperent Power flow in MVA
P = net.res_line.loc[line_index, "p_from_mw"]
Q = net.res_line.loc[line_index, "q_from_mvar"]

S = np.sqrt(P**2 + Q**2)

print("Apparent Power (MVA):", S)


#thermal Line limit in MVA
max_i = net.line.loc[line_index, "max_i_ka"]
vn = net.bus.loc[net.line.loc[line_index, "from_bus"], "vn_kv"]

s_limit = np.sqrt(3) * vn * max_i

print("Thermal Limit (MVA approx):", s_limit)


critical_lines = net.res_line.sort_values("loading_percent", ascending=False)

print(critical_lines[["loading_percent", "p_from_mw", "i_from_ka"]].head())




net.res_bus["vm_kv"] = net.res_bus["vm_pu"] * net.bus["vn_kv"]







print("\nLine Loading:")
print(net.res_line[["loading_percent"]])

print("\nTransformer Loading:")
print(net.res_trafo[["loading_percent"]])

print("Maximum Daily System Losses (MW):", df_study_results["losses_mw"].max())


# Renewable penetration %

renewable_generation = df_study_results["pv_mw"] + df_study_results["wind_mw"]
penetration = (renewable_generation / df_study_results["total_load_mw"] * 100).where(
    df_study_results["total_load_mw"] > 0,
    0.0,
)
print("\nMaximum Daily Renewable Penetration (%):", round(float(penetration.max()), 2))





# Most congested line

max_line_loading = net.res_line["loading_percent"].max()
worst_line = net.res_line["loading_percent"].idxmax()

print("\nMost Congested Line Index:", worst_line)
print("Max Line Loading (%):", round(max_line_loading, 2))
print("Line Name:", net.line.loc[worst_line, "name"])





# Voltage stability margin

min_voltage = net.res_bus["vm_pu"].min()
weak_bus = net.res_bus["vm_pu"].idxmin()

print("\nLowest Voltage (pu):", round(min_voltage, 4))
print("Weakest Bus:", net.bus.loc[weak_bus, "name"])





# Hosting capacity test

import copy

net_test = copy.deepcopy(net)

# Test a 25% wind over-nameplate scenario.
hosting_wind_mw = WIND_RATED_POWER_MW * 1.25
net_test.sgen.at[Wind_Gen, "p_mw"] = hosting_wind_mw

run_power_flow(net_test)

print(f"\nHosting Capacity Test (Wind = {hosting_wind_mw:.0f} MW)")
print("Max Line Loading (%):",
      round(net_test.res_line["loading_percent"].max(), 2))

print("Min Voltage (pu):",
      round(net_test.res_bus["vm_pu"].min(), 4))





# Exporting data to Excel for PSS/E-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#exporting data to Excel for PSS/E

import pandas as pd
import numpy as np
import os

# ==============================
# SYSTEM BASE
# ==============================
BASE_MVA = 100

# ==============================
# Save to Desktop Automatically
# ==============================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop, "psse_export.xlsx")

writer = pd.ExcelWriter(file_path, engine="xlsxwriter")

# ==============================
# BUS DATA
# ==============================

bus_df = pd.DataFrame({
    "Bus No": net.bus.index + 1,
    "Name": net.bus["name"],
    "Base kV": net.bus["vn_kv"],
    "Type": 1,
    "Vm (pu)": net.res_bus["vm_pu"],
    "Va (deg)": net.res_bus["va_degree"]
})

# Every external-grid bus is a PSS/E slack/swing bus in this network.
for slack_bus in net.ext_grid["bus"]:
    bus_df.loc[slack_bus, "Type"] = 3

bus_df.to_excel(writer, sheet_name="BUS", index=False)

# ==============================
# LOAD DATA
# ==============================

load_df = pd.DataFrame({
    "Bus No": net.load["bus"] + 1,
    "ID": 1,
    "P (MW)": net.load["p_mw"],
    "Q (MVAR)": net.load["q_mvar"]
})

load_df.to_excel(writer, sheet_name="LOAD", index=False)

# ==============================
# GENERATOR DATA
# ==============================

gen_df = pd.DataFrame({
    "Bus No": net.sgen["bus"] + 1,
    "ID": 1,
    "P (MW)": net.sgen["p_mw"],
    "Qmax": net.sgen["max_q_mvar"],
    "Qmin": net.sgen["min_q_mvar"],
    "Vset (pu)": 1.0
})

gen_df.to_excel(writer, sheet_name="GENERATOR", index=False)

# ==============================
# LINE DATA (Converted to PU)
# ==============================

line_data = []

for idx, row in net.line.iterrows():

    vn_kv = net.bus.loc[row["from_bus"], "vn_kv"]
    z_base = (vn_kv ** 2) / BASE_MVA

    r_total = row["r_ohm_per_km"] * row["length_km"]
    x_total = row["x_ohm_per_km"] * row["length_km"]
    rate_mva = np.sqrt(3) * vn_kv * row["max_i_ka"]

    r_pu = r_total / z_base
    x_pu = x_total / z_base

    line_data.append([
        row["from_bus"] + 1,
        row["to_bus"] + 1,
        r_pu,
        x_pu,
        0,
        rate_mva
    ])

line_df = pd.DataFrame(line_data,
    columns=["From Bus", "To Bus", "R (pu)", "X (pu)", "B (pu)", "Rate A (MVA)"]
)

line_df.to_excel(writer, sheet_name="BRANCH", index=False)

# ==============================
# TRANSFORMER DATA (PU)
# ==============================

trafo_data = []

for idx, row in net.trafo.iterrows():

    r_pu = row["vkr_percent"] / 100
    x_pu = np.sqrt((row["vk_percent"]/100)**2 - r_pu**2)

    trafo_data.append([
        row["hv_bus"] + 1,
        row["lv_bus"] + 1,
        r_pu,
        x_pu,
        row["sn_mva"],
        1.0
    ])

trafo_df = pd.DataFrame(trafo_data,
    columns=["From Bus", "To Bus", "R (pu)", "X (pu)", "Rating (MVA)", "Tap"]
)

trafo_df.to_excel(writer, sheet_name="TRANSFORMER", index=False)

writer.close()

print("PSS®E Excel file exported to Desktop successfully.")



#----------------------------- ANALYSIS AND VISUALIZATION -------------------------------------------------------------------------------------------------------------

# Graph time-window controls. Select a single day and plot only that 24-hour
# window. This keeps every chart focused on one operating day instead of a
# full week of stacked results.
SELECTED_DATE = WEATHER_DATE
selected_day = pd.Timestamp(SELECTED_DATE).normalize()
selected_day_start = selected_day + pd.Timedelta(minutes=1)
selected_day_end = selected_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
plot_output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "dlr_generation",
    SELECTED_DATE,
)
os.makedirs(plot_output_dir, exist_ok=True)
for old_plot in os.listdir(plot_output_dir):
    if old_plot.lower().endswith(".png"):
        os.remove(os.path.join(plot_output_dir, old_plot))
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") == "1"
if SHOW_PLOTS:
    plt.rcParams["figure.max_open_warning"] = 0


def format_daily_time_axis(axis):
    """Use a 24-hour clock axis from 12:01 AM to 11:59 PM for the selected day."""
    axis.set_xlim(selected_day_start, selected_day_end)
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    axis.tick_params(axis="x", rotation=45)


def save_generation_plot(filename):
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(
        os.path.join(plot_output_dir, filename),
        dpi=150,
        bbox_inches="tight",
    )
    if not SHOW_PLOTS:
        plt.close(figure)


lines = [line_id1, line_id2, line_id3, line_id4, line_id5]


for i in lines:
    print(net.line.name[i], "Loading (%) =", net.res_line.loading_percent[i])


selected_day_weather = df_weather[df_weather["time"].dt.normalize() == selected_day].copy()
if selected_day_weather.empty:
    raise ValueError(f"No weather records found for the selected date: {SELECTED_DATE}")

selected_day_time = selected_day_weather["time"]
selected_day_indexes = selected_day_weather.index.to_list()
selected_day_ampacity = [dlr_ampacity_profile[idx] for idx in selected_day_indexes]


df_hourly = pd.DataFrame(hourly_results)
report_results = df_hourly[df_hourly["time"].dt.normalize() == selected_day].copy()
if report_results.empty:
    raise ValueError(f"No hourly results found for the selected date: {SELECTED_DATE}")

selected_day_slr_loading = {
    line: [slr_line_loading_history[line][idx] for idx in selected_day_indexes]
    for line in lines
}
selected_day_dlr_loading = {
    line: [dlr_line_loading_history[line][idx] for idx in selected_day_indexes]
    for line in lines
}
selected_day_max_slr_loading = [
    max(selected_day_slr_loading[line][i] for line in lines)
    for i in range(len(selected_day_indexes))
]
selected_day_max_dlr_loading = [
    max(selected_day_dlr_loading[line][i] for line in lines)
    for i in range(len(selected_day_indexes))
]
selected_day_loading_difference = [
    selected_day_max_slr_loading[i] - selected_day_max_dlr_loading[i]
    for i in range(len(selected_day_indexes))
]


selected_day_generation = []
selected_day_load = []
selected_day_losses = []
selected_day_pv = []
selected_day_wind = []
selected_day_grid = []
selected_day_renewable_penetration = []

for idx in selected_day_indexes:
    profile_hour = idx % 24
    for load_id in load_profiles:
        net.load.at[load_id, "p_mw"] = load_profiles[load_id][profile_hour]
        net.load.at[load_id, "q_mvar"] = (
            load_profiles[load_id][profile_hour] * q_factor
        )
    net.sgen.at[PV_Gen, "p_mw"] = pv_generation_profile[profile_hour]
    net.sgen.at[Wind_Gen, "p_mw"] = wind_generation_profile[profile_hour]
    net.line.loc[lines, "max_i_ka"] = dlr_ampacity_profile[idx] / 1000
    run_power_flow(net)

    pv_generation = net.sgen.at[PV_Gen, "p_mw"]
    wind_generation = net.sgen.at[Wind_Gen, "p_mw"]
    renewable_generation = pv_generation + wind_generation
    grid_exchange = net.res_ext_grid["p_mw"].sum()
    generation = renewable_generation + grid_exchange
    load = net.load["p_mw"].sum()
    losses = net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum()
    renewable_penetration = (renewable_generation / load) * 100 if load > 0 else 0

    selected_day_generation.append(generation)
    selected_day_load.append(load)
    selected_day_losses.append(losses)
    selected_day_pv.append(pv_generation)
    selected_day_wind.append(wind_generation)
    selected_day_grid.append(grid_exchange)
    selected_day_renewable_penetration.append(renewable_penetration)


selected_day_grid_import = [max(value, 0) for value in selected_day_grid]
selected_day_grid_export = [max(-value, 0) for value in selected_day_grid]


def print_table(title, table, decimals=2):
    """Print compact, consistently rounded result tables."""
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(table.round(decimals).to_string(index=False))


def show_selected_day_summary():
    """Print the selected-date, 24-hour summary tables."""
    selected_offsets = list(range(min(24, len(report_results))))
    selected_results = report_results.iloc[selected_offsets].copy()
    selected_loading = selected_results[["time", "profile_hour", "max_slr_loading_percent", "max_dlr_loading_percent"]].rename(columns={
        "time": "Time", "profile_hour": "Profile hour",
        "max_slr_loading_percent": "Max SLR (%)", "max_dlr_loading_percent": "Max DLR (%)",
    })
    for line in lines:
        line_name = net.line.at[line, "name"]
        selected_loading[f"{line_name} DLR (%)"] = selected_results[f"{line_name} DLR (%)"].to_numpy()

    print_table(f"24-HOUR SYSTEM BALANCE - {SELECTED_DATE}", selected_results[[
        "time", "profile_hour", "total_load_mw", "pv_mw", "wind_mw",
        "grid_mw", "losses_mw", "min_voltage_pu",
    ]].rename(columns={
        "time": "Time", "profile_hour": "Profile hour",
        "total_load_mw": "Load (MW)", "pv_mw": "PV (MW)", "wind_mw": "Wind (MW)",
        "grid_mw": "Grid (MW)", "losses_mw": "Losses (MW)", "min_voltage_pu": "Min V (pu)",
    }))
    print_table(f"SELECTED OPERATING HOURS - LINE LOADING - {SELECTED_DATE}", selected_loading)


show_selected_day_summary()


system_summary = report_results[[
    "time", "profile_hour", "total_load_mw", "pv_mw", "wind_mw",
    "grid_mw", "losses_mw", "min_voltage_pu",
]].rename(columns={
    "time": "Time", "profile_hour": "Profile hour", "total_load_mw": "Load (MW)",
    "pv_mw": "PV (MW)", "wind_mw": "Wind (MW)", "grid_mw": "Grid (MW)",
    "losses_mw": "Losses (MW)", "min_voltage_pu": "Min V (pu)",
})
print_table("24-HOUR SYSTEM BALANCE", system_summary)

selected_offsets = [0, 6, 12, 18, 23]
selected_results = report_results.iloc[[
    offset for offset in selected_offsets if offset < len(report_results)
]].copy()
selected_loading = selected_results[["time", "profile_hour", "max_slr_loading_percent", "max_dlr_loading_percent"]].rename(columns={
    "time": "Time", "profile_hour": "Profile hour",
    "max_slr_loading_percent": "Max SLR (%)", "max_dlr_loading_percent": "Max DLR (%)",
})
for line in lines:
    line_name = net.line.at[line, "name"]
    selected_loading[f"{line_name} DLR (%)"] = selected_results[f"{line_name} DLR (%)"].to_numpy()
print_table(f"SELECTED OPERATING HOURS - LINE LOADING - {SELECTED_DATE}", selected_loading)

line_summary = []
for line in lines:
    line_name = net.line.at[line, "name"]
    slr_column = f"{line_name} SLR (%)"
    dlr_column = f"{line_name} DLR (%)"
    peak_row = df_hourly[dlr_column].idxmax()
    line_summary.append({
        "Line": line_name,
        "Peak time": df_hourly.at[peak_row, "time"],
        "Peak DLR (%)": df_hourly.at[peak_row, dlr_column],
        "Peak SLR (%)": df_hourly.at[peak_row, slr_column],
        "DLR loading range (%)": df_hourly[dlr_column].max() - df_hourly[dlr_column].min(),
    })
print_table("LINE LOADING SUMMARY", pd.DataFrame(line_summary))

most_loaded = max(line_summary, key=lambda item: item["Peak DLR (%)"])
print("\nNetwork observations")
print("- Highest DLR loading:", most_loaded["Line"])
print("- Highest DLR loading value (%):", round(most_loaded["Peak DLR (%)"], 2))
print("- DLR ampacity range (A):", round(min(dlr_ampacity_profile), 2), "to", round(max(dlr_ampacity_profile), 2))

# DLR-1 Weather Conditions: temperature, wind, cloud cover, and irradiance.
weather_irradiance = 1000.0 * np.maximum(
    np.sin(np.pi * (selected_day_weather["time"].dt.hour - 6) / 12), 0.0
) * (1.0 - 0.75 * selected_day_weather["cloud_cover_percent"] / 100.0)
fig, weather_axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
weather_series = [
    (selected_day_weather["temperature_c"], "Temperature (deg C)", PLOT_PALETTE["temperature"]),
    (selected_day_weather["wind_speed_ms"], "Wind Speed (m/s)", PLOT_PALETTE["wind_speed"]),
    (selected_day_weather["cloud_cover_percent"], "Cloud Cover (%)", PLOT_PALETTE["cloud_cover"]),
    (weather_irradiance, "Irradiance (W/m2)", PLOT_PALETTE["pv_generation"]),
]
for axis, (values, ylabel, color) in zip(weather_axes, weather_series):
    axis.plot(selected_day_time, values, color=color, linewidth=2.2, marker="o", markersize=3, label=ylabel)
    axis.set_ylabel(ylabel)
    axis.grid(True)
    axis.legend(loc="best")
format_daily_time_axis(weather_axes[-1])
weather_axes[-1].set_xlabel("Time")
fig.suptitle(f"DLR-1 Weather Conditions - {SELECTED_DATE}")
save_generation_plot("DLR-1_weather_conditions.png")

# DLR-2 SLR versus DLR ampacity.
plt.figure(figsize=(12, 6))
plt.plot(selected_day_time, [STATIC_AMPACITY] * len(selected_day_time), "--", color=PLOT_PALETTE["static_ampacity"], label="SLR Ampacity")
plt.plot(selected_day_time, selected_day_ampacity, color=PLOT_PALETTE["dlr_ampacity"], marker="o", label="DLR Ampacity")
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title(f"DLR-2 SLR vs DLR Line Ampacity - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("DLR-2_slr_vs_dlr_ampacity.png")

# DLR-3 capacity gain relative to the static rating.
capacity_gain = [(value - STATIC_AMPACITY) / STATIC_AMPACITY * 100 for value in selected_day_ampacity]
plt.figure(figsize=(12, 6))
plt.plot(selected_day_time, capacity_gain, color=PLOT_PALETTE["loading_difference"], marker="o", label="DLR Capacity Gain")
plt.axhline(0, color=PLOT_PALETTE["zero_reference"], linestyle="--")
plt.xlabel("Time"); plt.ylabel("Capacity Gain (%)")
plt.title(f"DLR-3 DLR Capacity Gain - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("DLR-3_capacity_gain.png")

# DLR-4 maximum line loading.
plt.figure(figsize=(12, 6))
plt.plot(selected_day_time, selected_day_max_slr_loading, color=PLOT_PALETTE["slr_loading"], marker="o", label="Maximum SLR Loading")
plt.plot(selected_day_time, selected_day_max_dlr_loading, color=PLOT_PALETTE["dlr_loading"], marker="o", label="Maximum DLR Loading")
plt.axhline(100, color=PLOT_PALETTE["congestion_limit"], linestyle="--", label="Congestion Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"DLR-4 Maximum Line Loading: SLR vs DLR - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("DLR-4_maximum_line_loading.png")

# Line-by-line SLR versus DLR loading in one graph.
plt.figure(figsize=(14, 7))
for line_number, line in enumerate(lines):
    line_name = net.line.loc[line, "name"]
    color = PLOT_COLORS[line_number % len(PLOT_COLORS)]
    plt.plot(selected_day_time, selected_day_slr_loading[line], color=color, linestyle="--", label=f"{line_name} SLR")
    plt.plot(selected_day_time, selected_day_dlr_loading[line], color=color, label=f"{line_name} DLR")
plt.axhline(100, color=PLOT_PALETTE["congestion_limit"], linestyle=":", label="Congestion Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"Line-by-Line SLR vs DLR Loading - {SELECTED_DATE}")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("line_by_line_slr_vs_dlr_loading.png")

# Separate SLR versus DLR loading comparison for each transmission line.
for line_number, line in enumerate(lines, start=1):
    line_name = net.line.at[line, "name"]
    line_slug = "_".join(line_name.lower().split())
    plt.figure(figsize=(12, 6))
    plt.plot(
        selected_day_time,
        selected_day_slr_loading[line],
        color=PLOT_PALETTE["slr_loading"],
        linestyle="--",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="SLR Loading",
    )
    plt.plot(
        selected_day_time,
        selected_day_dlr_loading[line],
        color=PLOT_PALETTE["dlr_loading"],
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="DLR Loading",
    )
    plt.axhline(
        100,
        color=PLOT_PALETTE["congestion_limit"],
        linestyle="--",
        label="Congestion Limit",
    )
    plt.xlabel("Time")
    plt.ylabel("Loading (%)")
    plt.title(f"Line {line_number} SLR vs DLR Loading - {SELECTED_DATE}\n{line_name}")
    plt.legend()
    plt.grid(True)
    format_daily_time_axis(plt.gca())
    save_generation_plot(f"line_{line_number}_{line_slug}_slr_vs_dlr.png")

# DLR-6 congestion comparison. Values above 100% are explicitly highlighted.
plt.figure(figsize=(12, 6))
plt.plot(selected_day_time, selected_day_max_slr_loading, color=PLOT_PALETTE["slr_loading"], marker="o", label="SLR Loading")
plt.plot(selected_day_time, selected_day_max_dlr_loading, color=PLOT_PALETTE["dlr_loading"], marker="o", label="DLR Loading")
plt.fill_between(selected_day_time, 100, selected_day_max_slr_loading, where=np.array(selected_day_max_slr_loading) > 100, color="#ef4444", alpha=0.25, label="SLR Congestion")
plt.fill_between(selected_day_time, 100, selected_day_max_dlr_loading, where=np.array(selected_day_max_dlr_loading) > 100, color="#f59e0b", alpha=0.25, label="DLR Congestion")
plt.axhline(100, color=PLOT_PALETTE["congestion_limit"], linestyle="--", label="100% Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"DLR-6 Congestion Under SLR vs DLR - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("DLR-6_congestion_slr_vs_dlr.png")

# DLR-7 is intentionally not plotted: ampacity changes the loading limit,
# not pandapower's active/reactive operating point, so SLR and DLR losses are identical here.
generation_history = selected_day_generation
load_history = selected_day_load
pv_history = selected_day_pv
wind_history = selected_day_wind
renewable_penetration_history = selected_day_renewable_penetration

# DLR-8 renewable generation and penetration using two y-axes.
fig, generation_axis = plt.subplots(figsize=(12, 6))
generation_axis.plot(selected_day_time, pv_history, color=PLOT_PALETTE["pv_generation"], marker="o", label="PV Generation (MW)")
generation_axis.plot(selected_day_time, wind_history, color=PLOT_PALETTE["wind_generation"], marker="o", label="Wind Generation (MW)")
generation_axis.plot(selected_day_time, np.array(pv_history) + np.array(wind_history), color=PLOT_PALETTE["total_generation"], linewidth=2.5, label="Total Renewable (MW)")
generation_axis.set_xlabel("Time"); generation_axis.set_ylabel("Generation (MW)")
penetration_axis = generation_axis.twinx()
penetration_axis.plot(selected_day_time, renewable_penetration_history, color=PLOT_PALETTE["renewable_penetration"], linestyle="--", marker="s", label="Renewable Penetration (%)")
penetration_axis.set_ylabel("Renewable Penetration (%)")
generation_axis.set_title(f"DLR-8 Renewable Generation & Penetration - {SELECTED_DATE}")
generation_axis.grid(True); format_daily_time_axis(generation_axis)
handles_1, labels_1 = generation_axis.get_legend_handles_labels()
handles_2, labels_2 = penetration_axis.get_legend_handles_labels()
generation_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
save_generation_plot("DLR-8_renewable_generation_penetration.png")

# Separate renewable-source profiles for easier inspection.
plt.figure(figsize=(12, 6))
plt.plot(
    selected_day_time,
    pv_history,
    color=PLOT_PALETTE["pv_generation"],
    linewidth=2.5,
    marker="o",
    label="PV Generation (MW)",
)
plt.xlabel("Time")
plt.ylabel("Generation (MW)")
plt.title(f"PV Generation Profile - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("PV_generation_profile.png")

plt.figure(figsize=(12, 6))
plt.plot(
    selected_day_time,
    wind_history,
    color=PLOT_PALETTE["wind_generation"],
    linewidth=2.5,
    marker="o",
    label="Wind Generation (MW)",
)
plt.xlabel("Time")
plt.ylabel("Generation (MW)")
plt.title(f"Wind Generation Profile - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_daily_time_axis(plt.gca())
save_generation_plot("wind_generation_profile.png")

print(f"Requested DLR graphs saved to: {plot_output_dir}")
if SHOW_PLOTS:
    plt.show()
