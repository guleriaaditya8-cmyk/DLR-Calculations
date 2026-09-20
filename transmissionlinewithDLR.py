"""24-hour PV/wind transmission study with static and dynamic line ratings.

The network parameters and profiles mirror ``Transmission_line.py``.  DLR is
applied to each 225 kV overhead line using the hourly ambient conditions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
from weather_api import load_open_meteo_weather, prompt_for_weather_date


# ---------------------------------------------------------------------------
# Weather and DLR model
# ---------------------------------------------------------------------------
STATIC_AMPACITY_A = 626.0  # matches the 225 kV line specification
MAX_DLR_AMPACITY_A = 2200.0
CONDUCTOR_DIAMETER_MM = 15.65
CONDUCTOR_TEMPERATURE_C = 80.0


def load_weather(selected_date=None) -> pd.DataFrame:
    """Load one 24-hour weather profile from Open-Meteo."""
    return load_open_meteo_weather(selected_date).rename(columns={
        "datetime": "time",
        "temperature_c": "temperature_2m",
    })


def calculate_ampacity_a(row: pd.Series) -> float:
    """Simplified heat-balance estimate, bounded by static/DLR limits."""
    ambient_c = row["temperature_2m"]
    wind_m_s = np.clip(row["wind_speed_100m"], 0.0, 10.0)
    cloud_percent = np.clip(row["cloud_cover"], 0.0, 100.0)
    delta_t = max(CONDUCTOR_TEMPERATURE_C - ambient_c, 0.0)
    resistance_ohm_m = (0.054 / 1000) * (1 + 0.00403 * (CONDUCTOR_TEMPERATURE_C - 20))
    solar_w_m2 = 1000 * (1 - 0.75 * cloud_percent / 100)
    solar_heat = 0.5 * solar_w_m2 * CONDUCTOR_DIAMETER_MM
    convective_cooling = 3.645 * wind_m_s**0.6 * delta_t
    radiative_cooling = 17.8 * (CONDUCTOR_DIAMETER_MM / 1000) * 0.6 * delta_t
    amps = np.sqrt(max((9 * convective_cooling + radiative_cooling - solar_heat) / resistance_ohm_m, 0.0))
    return float(np.clip(amps, STATIC_AMPACITY_A, MAX_DLR_AMPACITY_A))


# ---------------------------------------------------------------------------
# Network parameters: lines, transformers, loads, and generators
# ---------------------------------------------------------------------------
def add_transformer_types(net) -> None:
    for name, sn_mva, lv_kv, vk, vkr, pfe in [
        ("TR-1 210 MVA 225/15 kV", 210, 15, 12, 0.4, 8),
        ("TR-2 400 MVA 225/20 kV", 400, 20, 10, 0.3, 5),
        ("TR-3 231 MVA 225/20 kV", 231, 20, 10, 0.4, 8),
        ("TR-4 505 MVA 225/20 kV", 505, 20, 10, 0.3, 5),
        ("TR-5 336 MVA 225/20 kV", 336, 20, 10, 0.3, 5),
    ]:
        pp.create_std_type(net, {
            "sn_mva": sn_mva, "vn_hv_kv": 225, "vn_lv_kv": lv_kv,
            "vk_percent": vk, "vkr_percent": vkr, "pfe_kw": pfe,
            "i0_percent": 0.1, "shift_degree": 0,
        }, name=name, element="trafo")


def build_network():
    net = pp.create_empty_network()
    pp.create_std_type(net, {
        "c_nf_per_km": 12, "r_ohm_per_km": 0.025,
        "x_ohm_per_km": 0.32, "max_i_ka": STATIC_AMPACITY_A / 1000,
    }, name="225kV_line", element="line")
    add_transformer_types(net)

    buses = {name: pp.create_bus(net, kv, name=name) for name, kv in [
        ("PV plant", 15), ("PV 225 kV", 225), ("Zone 1 225 kV", 225),
        ("Zone 1 load", 20), ("Feeder grid 225 kV", 225),
        ("Zone 2 225 kV", 225), ("Zone 2 load", 20), ("Wind plant", 20),
        ("Wind 225 kV", 225), ("Zone 3 225 kV", 225), ("Zone 3 load", 20),
    ]}
    pp.create_ext_grid(net, buses["PV 225 kV"], vm_pu=1.02, name="Main grid")
    pp.create_ext_grid(net, buses["Feeder grid 225 kV"], vm_pu=1.02, name="Feeder grid")
    q_factor = np.tan(np.arccos(0.95))
    loads = [
        pp.create_load(net, buses["Zone 1 load"], 300, 300 * q_factor, name="Zone 1"),
        pp.create_load(net, buses["Zone 2 load"], 20, 20 * q_factor, name="Zone 2 A"),
        pp.create_load(net, buses["Zone 2 load"], 80, 80 * q_factor, name="Zone 2 B"),
        pp.create_load(net, buses["Zone 2 load"], 120, 120 * q_factor, name="Zone 2 C"),
        pp.create_load(net, buses["Zone 3 load"], 320, 320 * q_factor, name="Zone 3"),
    ]
    pv = pp.create_sgen(net, buses["PV plant"], 200, name="PV Generator", max_q_mvar=65.7, min_q_mvar=-65.7)
    wind = pp.create_sgen(net, buses["Wind plant"], 480, name="Wind Generator", max_q_mvar=154, min_q_mvar=-154)
    for hv, lv, typ, name in [
        ("PV 225 kV", "PV plant", "TR-1 210 MVA 225/15 kV", "TR-1 PV step-up"),
        ("Zone 1 225 kV", "Zone 1 load", "TR-2 400 MVA 225/20 kV", "TR-2 Zone 1"),
        ("Zone 2 225 kV", "Zone 2 load", "TR-3 231 MVA 225/20 kV", "TR-3 Zone 2"),
        ("Wind 225 kV", "Wind plant", "TR-4 505 MVA 225/20 kV", "TR-4 wind step-up"),
        ("Zone 3 225 kV", "Zone 3 load", "TR-5 336 MVA 225/20 kV", "TR-5 Zone 3"),
    ]:
        pp.create_transformer(net, buses[hv], buses[lv], typ, name=name)
    lines = [pp.create_line(net, buses[a], buses[b], km, "225kV_line", name=name) for a, b, km, name in [
        ("PV 225 kV", "Zone 1 225 kV", 35, "L1 PV to Zone 1"),
        ("PV 225 kV", "Zone 2 225 kV", 60, "L2 PV to Zone 2"),
        ("Feeder grid 225 kV", "Zone 2 225 kV", 60, "L3 feeder link"),
        ("Zone 2 225 kV", "Wind 225 kV", 60, "L4 feeder to wind"),
        ("Wind 225 kV", "Zone 3 225 kV", 60, "L5 wind to Zone 3"),
    ]]
    return net, {"loads": loads, "pv": pv, "wind": wind, "q_factor": q_factor}, lines


LOAD_SHAPE = [.45, .43, .42, .41, .41, .44, .50, .54, .57, .60, .61, .62, .63, .63, .64, .64, .65, .65, .65, .65, .63, .59, .53, .48]
PV_PROFILE_MW = [0, 0, 0, 0, 0, 0, 15, 45, 85, 125, 160, 185, 200, 195, 175, 140, 95, 45, 10, 0, 0, 0, 0, 0]
WIND_PROFILE_PU = [.74, .70, .68, .65, .62, .60, .58, .55, .52, .50, .48, .45, .43, .46, .50, .55, .62, .68, .72, .78, .80, .78, .76, .75]
NOMINAL_LOAD_MW = [300, 20, 80, 120, 320]


def run_study(weather: pd.DataFrame) -> pd.DataFrame:
    net, ids, lines = build_network()
    results = []
    for hour, (_, condition) in enumerate(weather.iterrows()):
        for load, nominal in zip(ids["loads"], NOMINAL_LOAD_MW):
            p_mw = nominal * LOAD_SHAPE[hour]
            net.load.loc[load, ["p_mw", "q_mvar"]] = [p_mw, p_mw * ids["q_factor"]]
        net.sgen.at[ids["pv"], "p_mw"] = PV_PROFILE_MW[hour]
        net.sgen.at[ids["wind"], "p_mw"] = 480 * WIND_PROFILE_PU[hour]
        net.line.loc[lines, "max_i_ka"] = STATIC_AMPACITY_A / 1000
        pp.runpp(net, algorithm="nr", init="dc", max_iteration=50)
        static = net.res_line.loc[lines, "loading_percent"].copy()
        net.line.loc[lines, "max_i_ka"] = condition["dlr_ampacity_a"] / 1000
        pp.runpp(net, algorithm="nr", init="results", max_iteration=50)
        dlr = net.res_line.loc[lines, "loading_percent"].copy()
        record = {"time": condition["time"], "hour": hour, "total_load_mw": net.load.p_mw.sum(),
                  "pv_mw": PV_PROFILE_MW[hour], "wind_mw": 480 * WIND_PROFILE_PU[hour],
                  "dlr_ampacity_a": condition["dlr_ampacity_a"], "grid_mw": net.res_ext_grid.p_mw.sum(),
                  "losses_mw": net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum(),
                  "min_voltage_pu": net.res_bus.vm_pu.min(),
                  "max_trafo_loading_percent": net.res_trafo.loading_percent.max()}
        for line in lines:
            name = net.line.at[line, "name"]
            record[f"{name} static_loading_percent"] = static.at[line]
            record[f"{name} dlr_loading_percent"] = dlr.at[line]
        results.append(record)
    return pd.DataFrame(results)


if __name__ == "__main__":
    weather_date = prompt_for_weather_date()
    weather = load_weather(weather_date)
    weather["dlr_ampacity_a"] = weather.apply(calculate_ampacity_a, axis=1).rolling(3, min_periods=1).mean()
    results = run_study(weather)
    results.to_csv("DLR_multiline_24h_results.csv", index=False)
    print(results[["hour", "total_load_mw", "pv_mw", "wind_mw", "dlr_ampacity_a", "grid_mw", "min_voltage_pu"]].round(2).to_string(index=False))
    fig, (profile_ax, loading_ax) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    profile_ax.plot(results.hour, results.total_load_mw, label="Load", linewidth=2)
    profile_ax.plot(results.hour, results.pv_mw, label="PV", linewidth=2)
    profile_ax.plot(results.hour, results.wind_mw, label="Wind", linewidth=2)
    profile_ax.set(ylabel="Power (MW)", title="24-hour load and generation profiles"); profile_ax.grid(); profile_ax.legend()
    for column in [c for c in results if c.endswith("dlr_loading_percent")]:
        loading_ax.plot(results.hour, results[column], label=column.removesuffix(" dlr_loading_percent"))
    loading_ax.axhline(100, color="crimson", linestyle="--", label="DLR limit")
    loading_ax.set(xlabel="Hour", ylabel="Line loading (%)", title="Dynamic line loading"); loading_ax.grid(); loading_ax.legend(ncol=2)
    plt.show()
