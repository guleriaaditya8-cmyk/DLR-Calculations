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
    create_switch,
    create_gen,
    create_sgen
)
from pandapower.run import runpp
from pandapower.auxiliary import LoadflowNotConverged
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
from weather_api import load_open_meteo_weather, prompt_for_weather_date

STATIC_AMPACITY_A = 626.0



# Load power factor used to derive the reactive component of every hourly load.
pf = 0.95

phi = np.arccos(pf)   # corrected
q_factor = np.tan(phi)

print("Reactive Power factor is:", q_factor)

p_mw = 300
q_mvar = p_mw * q_factor

print("Reactive Power is:", q_mvar)


# create empty net
net = pp.create_empty_network()


def run_power_flow(network):
    # The first hour uses a DC estimate; each following hour starts from the
    # preceding solution, which is much more stable for a time-series study.
    init = "results" if network.get("converged", False) else "dc"
    pp.runpp(network, algorithm="nr", init=init, max_iteration=50,
             tolerance_mva=1e-6)


def calculate_line_loading_percent(network, line, ampacity_a):
    """Calculate loading from the higher current at either line terminal."""
    terminal_current_ka = max(
        network.res_line.at[line, "i_from_ka"],
        network.res_line.at[line, "i_to_ka"],
    )
    return terminal_current_ka / (ampacity_a / 1000) * 100

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

# Grid interconnections at the PV and feeder 225 kV buses provide voltage and
# reactive-power support for the document-scale renewable plants and loads.
create_ext_grid(net, bus=b2, vm_pu=1.02, va_degree=0,
                name="Main 225 kV grid interconnection")
create_ext_grid(net, bus=b5, vm_pu=1.02, va_degree=0,
                name="Feeder 225 kV grid interconnection")

# capacitence for line RL is zero for transmission line and 0.1 for distribution line to simulate the effect of capacitance in distribution line and its effect on voltage profile and power flow of the system
pp.create_std_type(net, {
    "c_nf_per_km": 12,
    "r_ohm_per_km": 0.025,
    "x_ohm_per_km": 0.32,
    # The circuit specification uses a 0.626 kA static ampacity at 225 kV.
    "max_i_ka": STATIC_AMPACITY_A / 1000
}, name="225kV_line", element="line")



# create custom transformer standard types

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
}, name="TR-4 505 MVA 20/225 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 210,
    "vn_hv_kv": 225,
    "vn_lv_kv": 15,
    "vk_percent": 12,
    "vkr_percent": 0.4,
    "pfe_kw": 8,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-1 210 MVA 15/225 kV", element="trafo")

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
    "sn_mva": 336,
    "vn_hv_kv": 225,
    "vn_lv_kv": 20,
    "vk_percent": 10,
    "vkr_percent": 0.3,
    "pfe_kw": 5,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="TR-5 336 MVA 225/20 kV", element="trafo")



load_id1 = pp.create_load(net, bus=b4, p_mw=300, q_mvar=300 * q_factor, name="Zone 1 Load")
load_id2 = pp.create_load(net, bus=b7, p_mw=20, q_mvar=20 * q_factor, name="Zone 2 Load A")
load_id3 = pp.create_load(net, bus=b7, p_mw=80, q_mvar=80 * q_factor, name="Zone 2 Load B")
load_id4 = pp.create_load(net, bus=b7, p_mw=120, q_mvar=120 * q_factor, name="Zone 2 Load C")
load_id5 = pp.create_load(net, bus=b11, p_mw=320, q_mvar=320 * q_factor, name="Zone 3 Load")

# Zone 1 photovoltaic generation from the circuit specification.
PV_Gen = create_sgen(net, bus=b1, p_mw=200, max_q_mvar=65.7, min_q_mvar=-65.7, name="PV Generator 200 MW")

# Step up the PV plant from 15 kV to the 225 kV transmission system.
trfo_id1 = pp.create_transformer(net, hv_bus=b2, lv_bus=b1, std_type="TR-1 210 MVA 15/225 kV", name="TR-1 PV Step Up")

line_id1 = pp.create_line(net, from_bus=b2, to_bus=b3, length_km=35, name="Line 2 - 225 kV PV transmission",std_type="225kV_line")

trfo_id2 = pp.create_transformer(net, hv_bus=b3, lv_bus=b4, std_type="TR-2 400 MVA 225/20 kV", name="TR-2 Zone 1 Step Down")
# Interconnection between the PV and Zone 2 transmission buses.
line_id2 = pp.create_line(net, from_bus=b2, to_bus=b6, length_km=60, name="Line 4 - 225 kV PV to Zone 2",std_type="225kV_line")



# creating transmission line for feeder connction
line_id3 = pp.create_line(net, from_bus=b5, to_bus=b6, length_km=60, name="Line 3 - feeder transmission",std_type="225kV_line")
# creating transformers for step down from feeder to destribution
trfo_id3 = pp.create_transformer(net, hv_bus=b6, lv_bus=b7, std_type="TR-3 231 MVA 225/20 kV", name="TR-3 Zone 2 Step Down")

# Interconnection between the Zone 2 and wind transmission buses.
line_id4 = pp.create_line(net, from_bus=b6, to_bus=b9, length_km=60, name="Line 4 - 225 kV feeder to wind",std_type="225kV_line")






# Zone 3 wind generation from the circuit specification.
create_sgen(net, bus=b8, p_mw=480, max_q_mvar=154, min_q_mvar=-154, name="Wind Generator 480 MW")

# creating feeder bus connection to wind bus for step up transformer
 
trfo_id5 = pp.create_transformer(net, hv_bus=b9, lv_bus=b8, std_type="TR-4 505 MVA 20/225 kV", name="TR-4 Wind Step Up")

line_id5 = pp.create_line(net, from_bus=b9, to_bus=b10, length_km=60, name="Line 5 - 225 kV wind transmission",std_type="225kV_line")

trfo_id6 = pp.create_transformer(net, hv_bus=b10, lv_bus=b11, std_type="TR-5 336 MVA 225/20 kV", name="TR-5 Zone 3 Step Down")




#define load profile for each load to simulate the effect of load variation on the power flow and voltage profile of the system-------------------------------------------

# Per-unit daily demand shapes.  They scale the circuit-document nominal loads,
# avoiding the previous mismatch between the base load and profile values.
# Operating demand is held below the transformer nameplate ratings.  The
# nominal values below are installed/peak capacities, not simultaneous demand.
# Keep the operating demand within the voltage-stable region.  Congestion is
# created by the 0.626 kA line rating, not by forcing an infeasible load level.
load_shape = [0.45, 0.43, 0.42, 0.41, 0.41, 0.44, 0.50, 0.54,
              0.57, 0.60, 0.61, 0.62, 0.63, 0.73, 0.74, 0.74,
              0.72, 0.75, 0.70, 0.65, 0.63, 0.59, 0.53, 0.48]
nominal_load_mw = {
    load_id1: 300, load_id2: 20, load_id3: 80, load_id4: 120, load_id5: 320
}
load_profiles = {
    load_id: [nominal_mw * multiplier for multiplier in load_shape]
    for load_id, nominal_mw in nominal_load_mw.items()
}

# Huawei SUN2000-100KTL-M2 and Goldwind GW 136-4.8MW datasheet values.
PV_INVERTER_NOMINAL_AC_KW = 100.0
PV_INVERTER_MAX_AC_KW = 110.0
PV_INVERTER_COUNT = 2000
PV_INVERTER_EFFICIENCY = 0.986
PV_RATED_POWER_MW = PV_INVERTER_NOMINAL_AC_KW * PV_INVERTER_COUNT / 1000.0
PV_MAX_OUTPUT_MW = PV_INVERTER_MAX_AC_KW * PV_INVERTER_COUNT / 1000.0
PV_DC_ARRAY_RATED_MW = PV_RATED_POWER_MW / PV_INVERTER_EFFICIENCY

WIND_TURBINE_RATED_POWER_MW = 4.8
WIND_TURBINE_COUNT = 100
WIND_RATED_POWER_MW = WIND_TURBINE_RATED_POWER_MW * WIND_TURBINE_COUNT
WIND_HUB_HEIGHT_M = 110.0
WIND_CUT_IN_MS = 2.5
WIND_RATED_MS = 12.5
WIND_CUT_OUT_MS = 26.0
WEATHER_DATE = prompt_for_weather_date()


def load_weather_generation_inputs(selected_date):
    """Load the selected day's weather inputs from Open-Meteo."""
    return load_open_meteo_weather(selected_date)


def calculate_wind_power_mw(wind_speed_ms):
    """Apply the cut-in, rated, and cut-out wind-turbine power curve."""
    if wind_speed_ms < WIND_CUT_IN_MS or wind_speed_ms >= WIND_CUT_OUT_MS:
        return 0.0
    if wind_speed_ms >= WIND_RATED_MS:
        return WIND_RATED_POWER_MW
    normalized_speed = (
        (wind_speed_ms - WIND_CUT_IN_MS)
        / (WIND_RATED_MS - WIND_CUT_IN_MS)
    )
    return WIND_RATED_POWER_MW * normalized_speed ** 3


weather_generation = load_weather_generation_inputs(WEATHER_DATE)
WEATHER_DATE = weather_generation["datetime"].iloc[0].date().isoformat()
weather_hour = weather_generation["datetime"].dt.hour
solar_angle = np.maximum(np.sin(np.pi * (weather_hour - 6) / 12), 0.0)
clear_sky_irradiance = 1000.0 * solar_angle
cloud_factor = 1.0 - 0.75 * weather_generation["cloud_cover"] / 100.0
temperature_factor = 1.0 - 0.004 * (weather_generation["temperature_c"] - 25.0)
temperature_factor = temperature_factor.clip(lower=0.0)
solar_irradiance = clear_sky_irradiance * cloud_factor
pv_profile = (
    PV_DC_ARRAY_RATED_MW
    * (solar_irradiance / 1000.0)
    * temperature_factor
    * PV_INVERTER_EFFICIENCY
).clip(lower=0.0, upper=PV_MAX_OUTPUT_MW).round(2).tolist()
wind_speed_profile = weather_generation["wind_speed_100m"].tolist()
wind_profile = [
    calculate_wind_power_mw(wind_speed_ms) / WIND_RATED_POWER_MW
    for wind_speed_ms in wind_speed_profile
]
wind_generation_profile = [
    WIND_RATED_POWER_MW * wind_fraction
    for wind_fraction in wind_profile
]
wind_generation_status = []
for wind_speed_ms in wind_speed_profile:
    if wind_speed_ms < WIND_CUT_IN_MS:
        status = "below cut-in"
    elif wind_speed_ms >= WIND_CUT_OUT_MS:
        status = "above cut-out"
    elif wind_speed_ms >= WIND_RATED_MS:
        status = "rated output"
    else:
        status = "partial output"
    wind_generation_status.append(status)

print("PV generation range (MW):", min(pv_profile), "to", max(pv_profile))
print("Wind generation range (MW):", min(wind_generation_profile), "to", max(wind_generation_profile))
print("Wind generation hours by operating state:")
for status in ("below cut-in", "partial output", "rated output", "above cut-out"):
    print(f"  {status}: {wind_generation_status.count(status)}")


def set_interconnection_status(hour):
    """Keep every transmission line energized for every simulation hour."""
    net.line.loc[lines, "in_service"] = True
    return {
        "PV-feeder tie": "ON",
        "Feeder-grid tie": "ON",
    }


time = range(24)

lines = [line_id1, line_id2, line_id3, line_id4, line_id5]

line_loading_history = {line: [] for line in lines}
# DIAGNOSTIC: Keep one complete power-flow record for each profile hour.
hourly_results = []

for hour in range(24):

    # update loads
    for load_id in load_profiles:
        net.load.at[load_id, "p_mw"] = load_profiles[load_id][hour]
        net.load.at[load_id, "q_mvar"] = load_profiles[load_id][hour] * q_factor

    net.sgen.at[PV_Gen, "p_mw"] = pv_profile[hour]
    wind_generator = net.sgen.index[net.sgen["name"] == "Wind Generator 480 MW"][0]
    net.sgen.at[wind_generator, "p_mw"] = WIND_RATED_POWER_MW * wind_profile[hour]
    tie_status = set_interconnection_status(hour)

    # run power flow
    run_power_flow(net)

    # store line loading
    expected_total_load = sum(
        load_profiles[load_id][hour] for load_id in load_profiles
    )
    if not np.isclose(net.load["p_mw"].sum(), expected_total_load):
        raise RuntimeError(f"Load profile was not applied at hour {hour}")

    hourly_record = {
        "hour": hour,
        "total_load_mw": net.load["p_mw"].sum(),
        "total_pv_mw": net.sgen.loc[net.sgen["name"] == "PV Generator 200 MW", "p_mw"].sum(),
        "total_wind_mw": net.sgen.loc[net.sgen["name"] == "Wind Generator 480 MW", "p_mw"].sum(),
        "external_grid_mw": net.res_ext_grid["p_mw"].sum(),
        "system_losses_mw": net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum(),
        **tie_status,
    }

    for line in lines:
        line_name = net.line.loc[line, "name"]
        loading = calculate_line_loading_percent(net, line, STATIC_AMPACITY_A)
        pandapower_loading = net.res_line.loading_percent[line]
        if not np.isclose(loading, pandapower_loading, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"SLR loading mismatch on line {line}: "
                f"calculated {loading:.6f}%, "
                f"pandapower {pandapower_loading:.6f}%"
            )
        line_loading_history[line].append(loading)
        hourly_record[f"{line_name} | P_from_MW"] = net.res_line.p_from_mw[line]
        hourly_record[f"{line_name} | Q_from_MVAR"] = net.res_line.q_from_mvar[line]
        hourly_record[f"{line_name} | i_from_kA"] = net.res_line.i_from_ka[line]
        hourly_record[f"{line_name} | i_to_kA"] = net.res_line.i_to_ka[line]
        hourly_record[f"{line_name} | loading_percent"] = loading

    for transformer in net.trafo.index:
        transformer_name = net.trafo.at[transformer, "name"]
        hourly_record[f"{transformer_name} | loading_percent"] = (
            net.res_trafo.loading_percent[transformer]
        )

    for bus in net.bus.index:
        bus_name = net.bus.at[bus, "name"]
        hourly_record[f"{bus_name} | voltage_pu"] = net.res_bus.vm_pu[bus]

    hourly_results.append(hourly_record)

hourly_results_df = pd.DataFrame(hourly_results)
line_loading_columns = [
    f"{net.line.at[line, 'name']} | loading_percent" for line in lines
]
trafo_loading_columns = [
    f"{net.trafo.at[transformer, 'name']} | loading_percent"
    for transformer in net.trafo.index
]
bus_voltage_columns = [
    f"{net.bus.at[bus, 'name']} | voltage_pu" for bus in net.bus.index
]





# run power flow to get the results for the initial load profile-----------------------------------------------------------------------------------------------



run_power_flow(net)

#simulating load flow of each transmission line
net.res_line

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

print("\n--- Loading ---")
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

print("Maximum System Losses (MW):", hourly_results_df["system_losses_mw"].max())
print("Maximum Daily Line Loading:", hourly_results_df[line_loading_columns].max().max())
print("Maximum Daily Transformer Loading:", hourly_results_df[trafo_loading_columns].max().max())
print("Minimum Daily Line Loading:", hourly_results_df[line_loading_columns].min().min())
print("Minimum Daily Transformer Loading:", hourly_results_df[trafo_loading_columns].min().min())


# Renewable penetration %

renewable_generation = hourly_results_df["total_pv_mw"] + hourly_results_df["total_wind_mw"]
penetration = (
    renewable_generation / hourly_results_df["total_load_mw"] * 100
).where(hourly_results_df["total_load_mw"] > 0, 0.0)

print("Maximum Daily Renewable Penetration (%):", round(float(penetration.max()), 2))





# Most congested line

max_line_loading = hourly_results_df[line_loading_columns].max().max()
worst_column = hourly_results_df[line_loading_columns].max().idxmax()
worst_line = next(line for line in lines if f"{net.line.at[line, 'name']} | loading_percent" == worst_column)

print("\nMost Congested Line Index:", worst_line)
print("Max Line Loading (%):", round(max_line_loading, 2))
print("Line Name:", net.line.loc[worst_line, "name"])





# Voltage stability margin

min_voltage = hourly_results_df[bus_voltage_columns].min().min()
weak_column = hourly_results_df[bus_voltage_columns].min().idxmin()
weak_bus = next(bus for bus in net.bus.index if f"{net.bus.at[bus, 'name']} | voltage_pu" == weak_column)

print("\nLowest Voltage (pu):", round(min_voltage, 4))
print("Weakest Bus:", net.bus.loc[weak_bus, "name"])





# N-1 contingency test is disabled because this study keeps every line in service.

import copy

net_n1 = copy.deepcopy(net)
net_n1.line.loc[lines, "in_service"] = True
print("\nN-1 Test skipped: all transmission lines remain in service.")






# Hosting capacity test

net_test = copy.deepcopy(net)

# Test a 25% wind over-nameplate scenario using the actual generator index.
hosting_wind_mw = WIND_RATED_POWER_MW * 1.25
wind_generator = net.sgen.index[net.sgen["name"] == "Wind Generator 480 MW"][0]
net_test.sgen.at[wind_generator, "p_mw"] = hosting_wind_mw

try:
    run_power_flow(net_test)
    print(f"\nHosting Capacity Test (Wind = {hosting_wind_mw:.0f} MW)")
    print("Max Line Loading (%):",
          round(net_test.res_line["loading_percent"].max(), 2))
    print("Min Voltage (pu):",
          round(net_test.res_bus["vm_pu"].min(), 4))
except LoadflowNotConverged:
    print("\nHosting Capacity Test: wind increase is not feasible.")





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
os.makedirs(desktop, exist_ok=True)
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

    r_pu = r_total / z_base
    x_pu = x_total / z_base

    line_data.append([
        row["from_bus"] + 1,
        row["to_bus"] + 1,
        r_pu,
        x_pu,
        0,
        np.sqrt(3) * vn_kv * row["max_i_ka"]
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



lines = [line_id1, line_id2, line_id3, line_id4, line_id5]

plt.rcParams.update({
    "figure.facecolor": "#f7f9fc",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#1e293b",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#dbe3ee",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cbd5e1",
    "font.size": 10,
})

plot_colors = plt.get_cmap("tab10").colors



# DIAGNOSTIC: All tables and plots below use the profile-driven results collected above.
df_hourly = pd.DataFrame(hourly_results)
df_hourly["renewable_generation_mw"] = (
    df_hourly["total_pv_mw"] + df_hourly["total_wind_mw"]
)
df_hourly["renewable_penetration_percent"] = (
    df_hourly["renewable_generation_mw"] / df_hourly["total_load_mw"] * 100
).where(df_hourly["total_load_mw"] > 0, 0.0)

SELECTED_DATE = WEATHER_DATE
selected_day = pd.Timestamp(SELECTED_DATE).normalize()
plot_time = pd.date_range(selected_day, periods=len(df_hourly), freq="h")
selected_day_start = selected_day + pd.Timedelta(minutes=1)
selected_day_end = selected_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
plot_output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "transmission_line",
    SELECTED_DATE,
)
os.makedirs(plot_output_dir, exist_ok=True)
# Remove stale charts from an earlier run of the same selected date.
for old_plot in os.listdir(plot_output_dir):
    if old_plot.lower().endswith(".png"):
        os.remove(os.path.join(plot_output_dir, old_plot))
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") == "1"
if SHOW_PLOTS:
    plt.rcParams["figure.max_open_warning"] = 0


def format_selected_day_axis(axis):
    """Use a 12:01 AM to 11:59 PM clock range on every result graph."""
    axis.set_xlim(selected_day_start, selected_day_end)
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    axis.tick_params(axis="x", rotation=45)


def save_current_plot(filename):
    """Save each daily graph so it remains available without a GUI plot window."""
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(
        os.path.join(plot_output_dir, filename),
        dpi=150,
        bbox_inches="tight",
    )
    if not SHOW_PLOTS:
        plt.close(figure)


line_labels = {
    line_id1: "L2 PV transmission",
    line_id2: "L4 PV to Zone 2",
    line_id3: "L3 feeder link",
    line_id4: "L4 feeder to wind",
    line_id5: "L5 wind transmission",
}


def print_table(title, table, decimals=2):
    """Print compact, consistently rounded console tables."""
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(table.round(decimals).to_string(index=False))


system_summary = df_hourly[[
    "hour", "total_load_mw", "total_pv_mw", "total_wind_mw",
    "external_grid_mw", "system_losses_mw",
]].rename(columns={
    "hour": "Hour",
    "total_load_mw": "Load (MW)",
    "total_pv_mw": "PV (MW)",
    "total_wind_mw": "Wind (MW)",
    "external_grid_mw": "Grid (MW)",
    "system_losses_mw": "Losses (MW)",
})
print_table("24-HOUR SYSTEM BALANCE", system_summary)

selected_hours = [0, 6, 12, 18, 19, 23]
selected_summary = system_summary[system_summary["Hour"].isin(selected_hours)].copy()
selected_summary["PV-feeder tie"] = df_hourly.loc[
    df_hourly["hour"].isin(selected_hours), "PV-feeder tie"
].to_numpy()
selected_summary["Feeder-grid tie"] = df_hourly.loc[
    df_hourly["hour"].isin(selected_hours), "Feeder-grid tie"
].to_numpy()
for line in lines:
    line_name = net.line.loc[line, "name"]
    selected_summary[f"{line_labels[line]} (%)"] = df_hourly.loc[
        df_hourly["hour"].isin(selected_hours),
        f"{line_name} | loading_percent",
    ].to_numpy()
print_table("SELECTED OPERATING HOURS — LINE LOADING", selected_summary)

# DIAGNOSTIC: Compare actual variation in each line's flow, current, and loading.
line_variation = []
for line in lines:
    line_name = net.line.loc[line, "name"]
    line_variation.append({
        "Line": line_labels[line],
        "Peak hour": int(df_hourly[f"{line_name} | loading_percent"].idxmax()),
        "Peak loading (%)": df_hourly[f"{line_name} | loading_percent"].max(),
        "Peak current (kA)": max(
            df_hourly[f"{line_name} | i_from_kA"].max(),
            df_hourly[f"{line_name} | i_to_kA"].max(),
        ),
        "P_from_range_MW": (
            df_hourly[f"{line_name} | P_from_MW"].max()
            - df_hourly[f"{line_name} | P_from_MW"].min()
        ),
        "Q_from_range_MVAR": (
            df_hourly[f"{line_name} | Q_from_MVAR"].max()
            - df_hourly[f"{line_name} | Q_from_MVAR"].min()
        ),
        "i_from_range_kA": (
            df_hourly[f"{line_name} | i_from_kA"].max()
            - df_hourly[f"{line_name} | i_from_kA"].min()
        ),
        "loading_range_percent": (
            df_hourly[f"{line_name} | loading_percent"].max()
            - df_hourly[f"{line_name} | loading_percent"].min()
        ),
        "max_loading_percent": df_hourly[
            f"{line_name} | loading_percent"
        ].max(),
    })

line_variation_df = pd.DataFrame(line_variation)
line_summary = line_variation_df[[
    "Line", "Peak hour", "Peak loading (%)", "Peak current (kA)",
    "loading_range_percent",
]].rename(columns={"loading_range_percent": "Loading range (%)"})
print_table("LINE LOADING SUMMARY", line_summary)
most_affected_line = line_variation_df.loc[
    line_variation_df["loading_range_percent"].idxmax(), "Line"
]
most_loaded_line = line_variation_df.loc[
    line_variation_df["Peak loading (%)"].idxmax(), "Line"
]
print("\nNetwork observations")
print("• Largest loading variation:", most_affected_line)
print("• Highest peak loading:", most_loaded_line)
print(
    "Fixed PV generation (MW):",
    df_hourly["total_pv_mw"].nunique() == 1,
    "| Fixed wind generation (MW):",
    df_hourly["total_wind_mw"].nunique() == 1,
)
print(
    "• External-grid balancing range (MW):",
    round(df_hourly["external_grid_mw"].min(), 2), "to",
    round(df_hourly["external_grid_mw"].max(), 2),
)

# ==============================
# REQUESTED SLR FIGURES 4.1-4.7
# ==============================
slr_plot_time = plot_time
slr_load = df_hourly["total_load_mw"]
slr_pv = df_hourly["total_pv_mw"]
slr_wind = df_hourly["total_wind_mw"]
slr_grid = df_hourly["external_grid_mw"]
slr_losses = df_hourly["system_losses_mw"]

# Figure 4.1 - Load and Renewable Generation Profile.
plt.figure(figsize=(12, 6))
plt.plot(slr_plot_time, slr_load, color="#264653", linewidth=2.4, label="Load (MW)")
plt.plot(slr_plot_time, slr_pv, color="#e9c46a", linewidth=2.4, label="PV (MW)")
plt.plot(slr_plot_time, slr_wind, color="#457b9d", linewidth=2.4, label="Wind (MW)")
plt.plot(slr_plot_time, slr_grid, color="#7b2cbf", linewidth=2.4, label="Grid Exchange (MW)")
plt.xlabel("Time"); plt.ylabel("Power (MW)")
plt.title(f"Figure 4.1 - Load and Renewable Generation Profile - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_1_load_renewable_generation.png")

# Figure 4.2 - Transmission Line Loading Under SLR.
plt.figure(figsize=(13, 7))
for line_number, line in enumerate(lines, start=1):
    line_name = net.line.at[line, "name"]
    plt.plot(
        slr_plot_time,
        df_hourly[f"{line_name} | loading_percent"],
        color=plot_colors[(line_number - 1) % len(plot_colors)],
        linewidth=2.1,
        label=f"L{line_number} - {line_name}",
    )
plt.axhline(100, color="#dc2626", linestyle="--", linewidth=2, label="100% Congestion Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"Figure 4.2 - Transmission Line Loading Under SLR - {SELECTED_DATE}")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_2_transmission_loading_slr.png")

# Figure 4.3 - Individual Transmission-Line Loading Under SLR.
fig, line_axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
for line_number, (axis, line) in enumerate(zip(line_axes, lines), start=1):
    line_name = net.line.at[line, "name"]
    axis.plot(
        slr_plot_time,
        df_hourly[f"{line_name} | loading_percent"],
        color=plot_colors[(line_number - 1) % len(plot_colors)],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=f"L{line_number} - {line_name}",
    )
    axis.axhline(100, color="#dc2626", linestyle="--", label="100% Limit")
    axis.set_ylabel("Loading (%)")
    axis.legend(loc="upper left")
    axis.grid(True)
format_selected_day_axis(line_axes[-1]); line_axes[-1].set_xlabel("Time")
fig.suptitle(f"Figure 4.3 - Individual Transmission-Line Loading Under SLR - {SELECTED_DATE}")
save_current_plot("Figure_4_3_individual_line_loading_slr.png")

# Figure 4.4 - Transformer Loading Under SLR.
plt.figure(figsize=(13, 7))
for transformer_number, transformer in enumerate(net.trafo.index, start=1):
    transformer_name = net.trafo.at[transformer, "name"]
    plt.plot(
        slr_plot_time,
        df_hourly[f"{transformer_name} | loading_percent"],
        color=plot_colors[(transformer_number - 1) % len(plot_colors)],
        linewidth=2.0,
        label=transformer_name,
    )
plt.axhline(100, color="#dc2626", linestyle="--", linewidth=2, label="100% Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"Figure 4.4 - Transformer Loading Under SLR - {SELECTED_DATE}")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_4_transformer_loading_slr.png")

# Figure 4.5 - Bus Voltage Profile Under SLR.
plt.figure(figsize=(13, 7))
for bus_number, bus in enumerate(net.bus.index, start=1):
    bus_name = net.bus.at[bus, "name"]
    plt.plot(
        slr_plot_time,
        df_hourly[f"{bus_name} | voltage_pu"],
        color=plot_colors[(bus_number - 1) % len(plot_colors)],
        linewidth=1.7,
        label=bus_name,
    )
plt.axhline(0.95, color="#f59e0b", linestyle="--", label="0.95 pu Lower Guide")
plt.axhline(1.05, color="#dc2626", linestyle="--", label="1.05 pu Upper Guide")
plt.xlabel("Time"); plt.ylabel("Voltage (pu)")
plt.title(f"Figure 4.5 - Bus Voltage Profile Under SLR - {SELECTED_DATE}")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_5_bus_voltage_slr.png")

# Figure 4.6 - Transmission Losses Under SLR.
plt.figure(figsize=(12, 6))
plt.plot(slr_plot_time, slr_losses, color="#f9844a", linewidth=2.5, marker="o", label="Transmission and Transformer Losses")
plt.xlabel("Time"); plt.ylabel("Losses (MW)")
plt.title(f"Figure 4.6 - Transmission Losses Under SLR - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_6_transmission_losses_slr.png")

# Figure 4.7 - SLR Congestion Summary.
max_slr_by_hour = df_hourly[[
    f"{net.line.at[line, 'name']} | loading_percent" for line in lines
]].max(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(slr_plot_time, max_slr_by_hour, color="#d1495b", linewidth=2.5, marker="o", label="Maximum SLR Loading")
plt.fill_between(slr_plot_time, 100, max_slr_by_hour, where=max_slr_by_hour > 100, color="#ef4444", alpha=0.28, label="Congested Period")
plt.axhline(100, color="#dc2626", linestyle="--", linewidth=2, label="100% Congestion Limit")
plt.xlabel("Time"); plt.ylabel("Loading (%)")
plt.title(f"Figure 4.7 - SLR Congestion Summary - {SELECTED_DATE}")
plt.legend(); plt.grid(True); format_selected_day_axis(plt.gca())
save_current_plot("Figure_4_7_slr_congestion_summary.png")

print(f"Requested SLR Figures 4.1-4.7 saved to: {plot_output_dir}")

# SLR line-loading comparison plot following the DLR code styling.
max_slr_loading = [
    max(
        df_hourly.loc[idx, f"{net.line.loc[line, 'name']} | loading_percent"]
        for line in lines
        if pd.notna(df_hourly.loc[idx, f"{net.line.loc[line, 'name']} | loading_percent"])
    )
    for idx in range(len(df_hourly))
]

plt.figure(figsize=(12, 6))
for line in lines:
    line_name = net.line.loc[line, "name"]
    color = plot_colors[lines.index(line) % len(plot_colors)]
    plt.plot(plot_time, df_hourly[f"{line_name} | loading_percent"], color=color,
             linewidth=2.2, marker="o", markersize=3.5, label=line_name)
plt.axhline(80, color="#f59e0b", linestyle="--", label="Warning Limit")
plt.axhline(100, color="#dc2626", linestyle="--", label="Congestion Limit")
plt.xlabel("Date and Time")
plt.ylabel("Line Loading (%)")
plt.title(f"Static Line Rating (SLR) Loading by Circuit - {selected_day:%Y-%m-%d}")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("01_slr_loading_by_circuit.png")

# Individual line-loading plots. Each transmission line gets its own graph so
# its loading profile can be inspected without the other circuits overlapping it.
for line_number, line in enumerate(lines, start=1):
    line_name = net.line.loc[line, "name"]
    line_loading = df_hourly[f"{line_name} | loading_percent"]
    line_slug = "_".join(line_name.lower().split())

    plt.figure(figsize=(12, 6))
    plt.plot(
        plot_time,
        line_loading,
        color=plot_colors[(line_number - 1) % len(plot_colors)],
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        label=f"{line_name} Loading",
    )
    plt.axhline(80, color="#f59e0b", linestyle="--", label="Warning Limit")
    plt.axhline(100, color="#dc2626", linestyle="--", label="Congestion Limit")
    plt.xlabel("Date and Time")
    plt.ylabel("Line Loading (%)")
    plt.title(f"{line_name} - SLR Line Loading - {selected_day:%Y-%m-%d}")
    plt.legend()
    plt.grid(True)
    format_selected_day_axis(plt.gca())
    save_current_plot(f"line_{line_number}_{line_slug}_loading.png")

# PV generation is a production profile, not a line-loading measurement.
# Plot it separately so its daytime shape is not confused with network flow.
plt.figure(figsize=(12, 6))
plt.plot(
    plot_time,
    df_hourly["total_pv_mw"],
    color="#e9c46a",
    linewidth=2.5,
    marker="o",
    markersize=3.5,
    label="PV Generation",
)
plt.xlabel("Date and Time")
plt.ylabel("PV Generation (MW)")
plt.title(f"PV Generation Profile - {selected_day:%Y-%m-%d}")
plt.legend()
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("06_pv_generation_profile.png")

# Wind generation profile driven by the measured 100 m wind speed.
plt.figure(figsize=(12, 6))
plt.plot(
    plot_time,
    df_hourly["total_wind_mw"],
    color="#457b9d",
    linewidth=2.5,
    marker="o",
    markersize=3.5,
    label="Wind Generation",
)
plt.xlabel("Date and Time")
plt.ylabel("Wind Generation (MW)")
plt.title(f"Weather-Driven Wind Generation Profile - {selected_day:%Y-%m-%d}")
plt.legend()
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("07_wind_generation_profile.png")

# Dedicated circuit-wide static load plot.
plt.figure(figsize=(12, 6))
plt.plot(plot_time, max_slr_loading, color="#2563eb", linewidth=2.5,
         marker="o", markersize=3.5, label="Maximum SLR Loading")
plt.axhline(80, color="#f59e0b", linestyle="--", label="Warning Limit")
plt.axhline(100, color="#dc2626", linestyle="--", label="Congestion Limit")
plt.xlabel("Date and Time")
plt.ylabel("Loading (%)")
plt.title(f"Maximum SLR Line Loading - {selected_day:%Y-%m-%d}")
plt.legend()
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("02_maximum_slr_loading.png")

# System power balance plot.
generation_history = (
    df_hourly["total_pv_mw"] + df_hourly["total_wind_mw"] + df_hourly["external_grid_mw"]
)
load_history = df_hourly["total_load_mw"]
loss_history = df_hourly["system_losses_mw"]

plt.figure(figsize=(12, 6))
plt.plot(plot_time, generation_history, color="#16a34a", linewidth=2.5,
         marker="o", markersize=3.5, label="Total Generation")
plt.plot(plot_time, load_history, color="#2563eb", linewidth=2.5,
         marker="o", markersize=3.5, label="Total Load")
plt.plot(plot_time, loss_history, color="#f97316", linewidth=2.5,
         marker="o", markersize=3.5, label="System Losses")
plt.xlabel("Date and Time")
plt.ylabel("Power (MW)")
plt.title(f"Smart Grid Power Balance - {selected_day:%Y-%m-%d}")
plt.legend(loc="upper left")
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("03_system_power_balance.png")

# Grid import/export plot.
grid_import_history = [max(value, 0) for value in df_hourly["external_grid_mw"]]
grid_export_history = [max(-value, 0) for value in df_hourly["external_grid_mw"]]

plt.figure(figsize=(12, 6))
plt.plot(plot_time, grid_import_history, color="#277da1", linewidth=2.5,
         marker="o", markersize=3.5, label="Grid Import")
plt.plot(plot_time, grid_export_history, color="#43aa8b", linewidth=2.5,
         marker="o", markersize=3.5, label="Grid Export")
plt.fill_between(plot_time, grid_export_history, color="#43aa8b", alpha=0.10)
plt.xlabel("Date and Time")
plt.ylabel("Power (MW)")
plt.title("Net Grid Import and Renewable-Surplus Export")
plt.legend()
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("04_grid_import_export.png")

# Renewable penetration plot.
plt.figure(figsize=(12, 6))
plt.plot(plot_time, df_hourly["renewable_penetration_percent"], color="#90be6d",
         linewidth=2.5, marker="o", markersize=3.5, label="Renewable Penetration")
plt.axhline(100, color="#6d6875", linestyle="--", label="100% of Load")
plt.xlabel("Date and Time")
plt.ylabel("Renewable Penetration (%)")
plt.title(f"Renewable Penetration Across the Feeder - {selected_day:%Y-%m-%d}")
plt.legend()
plt.grid(True)
format_selected_day_axis(plt.gca())
save_current_plot("05_renewable_penetration.png")

print(f"Daily graphs saved to: {plot_output_dir}")
if SHOW_PLOTS:
    plt.show()
