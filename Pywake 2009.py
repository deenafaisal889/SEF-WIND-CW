
# ============================================================
# OFFSHORE WIND FARM - ANGLESEY / WYLFA
# SEF01 WIND ENERGY COURSEWORK 2026-27
#
# WIND DATA: Valley 2009
# MAIN WIND DIRECTION: SOUTH (180 degrees)
# WIND TRAVELS: SOUTH TO NORTH
#
# TURBINE OPTIONS:
# Vestas V164 7 MW:              13 turbines = 91 MW
# Siemens Gamesa SG10 10 MW:      9 turbines = 90 MW
# Vestas V236 15 MW:              6 turbines = 90 MW
#
# LAYOUTS:
# 1. Linear / aligned
# 2. Staggered
#
# WAKE MODEL: Jensen (NOJ)
#
# GRAPHS ARE DISPLAYED ONLY.
# NO AUTOMATIC SAVING OF GRAPHS OR FILES.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from pathlib import Path

from py_wake import NOJ, HorizontalGrid
from py_wake.site import UniformSite
from py_wake.site.shear import PowerShear
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

FOLDER = Path.home() / "Downloads"

WIND_FILE = (
    FOLDER / "valley_2009_cleaned_for_pywake.csv"
)

CURVE_FILES = {
    "V164": FOLDER / "Vestas_V164_7000_power_ct.csv",
    "SG10": FOLDER / "Siemens_Gamesa_SG10_193_power_ct.csv",
    "V236": FOLDER / "Vestas_V236_15000_power_ct.csv"
}

# Check that all files exist

for file in [WIND_FILE, *CURVE_FILES.values()]:

    if not file.exists():

        raise FileNotFoundError(
            f"File not found: {file}\n"
            "Check that all CSV files are in Downloads."
        )


# ============================================================
# 3. TURBINE INFORMATION
# ============================================================

TURBINES = {

    "V164": {
        "name": "Vestas V164 7 MW",
        "diameter": 164,
        "hub_height": 105,
        "rated_power": 7,
        "cut_out": 25,  # m/s, assumed manufacturer cut-out for this V164 variant
        "number": 13
    },

    "SG10": {
        "name": "Siemens Gamesa SG 10.0-193 DD",
        "diameter": 193,
        "hub_height": 140,
        "rated_power": 10,
        "cut_out": 25,  # m/s, using the 25 m/s specification selected for this study
        "number": 9
    },

    "V236": {
        "name": "Vestas V236 15 MW",
        "diameter": 236,
        "hub_height": 145,
        "rated_power": 15,
        "cut_out": 30,  # m/s, conservative end of the 30-31 m/s range in the supplied specification
        "number": 6
    }
}


# ============================================================
# 4. MODEL ASSUMPTIONS
# ============================================================

REFERENCE_HEIGHT = 10
SHEAR_EXPONENT = 0.11

# Wind shear is applied by PyWake from 10 m to each turbine hub height.
# Cut-out speeds are specified separately for each turbine in TURBINES.
# Note: the supplied power curves end at 25 m/s

TURBULENCE_INTENSITY = 0.10

# Turbine spacing in rotor diameters
CROSSWIND_SPACING = 5
DOWNWIND_SPACING = 7

# Dominant southerly wind used to orient the layouts
LAYOUT_WIND_DIRECTION = 180

# Conditions used for the illustrative wake maps
WAKE_WIND_DIRECTION = 180
WAKE_WIND_SPEED = 9.8

# Wake map resolution
WAKE_GRID_POINTS = 80


# ============================================================
# 5. LOAD AND CLEAN VALLEY WIND DATA
# ============================================================

wind = pd.read_csv(WIND_FILE)

required_columns = [
    "wind_speed_ms",
    "wind_direction_deg_from",
    "valid_speed_and_direction"
]

for column in required_columns:

    if column not in wind.columns:

        raise ValueError(
            f"Missing column in wind data: {column}"
        )


# Keep valid wind observations

valid_flag = (
    wind["valid_speed_and_direction"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

wind = wind[valid_flag].copy()

wind["wind_speed_ms"] = pd.to_numeric(
    wind["wind_speed_ms"],
    errors="coerce"
)

wind["wind_direction_deg_from"] = pd.to_numeric(
    wind["wind_direction_deg_from"],
    errors="coerce"
)

wind = wind.dropna(
    subset=[
        "wind_speed_ms",
        "wind_direction_deg_from"
    ]
)

wind = wind[
    (wind["wind_speed_ms"] >= 0) &
    (wind["wind_direction_deg_from"] >= 0) &
    (wind["wind_direction_deg_from"] <= 360)
].copy()

ws = wind["wind_speed_ms"].to_numpy(dtype=float)

wd = (
    wind["wind_direction_deg_from"].to_numpy(dtype=float)
    % 360
)

if len(ws) == 0:

    raise ValueError(
        "No valid wind observations found."
    )


print("\n========== WIND DATA ==========")

print("Valid observations:", len(ws))

print(
    "Mean wind speed:",
    round(np.mean(ws), 2),
    "m/s"
)

print(
    "Maximum wind speed:",
    round(np.max(ws), 2),
    "m/s"
)


# ============================================================
# 6. CALCULATE CIRCULAR MEAN WIND DIRECTION
# ============================================================

wd_rad = np.deg2rad(wd)

mean_sin = np.mean(np.sin(wd_rad))
mean_cos = np.mean(np.cos(wd_rad))

prevailing_wd = (
    np.degrees(
        np.arctan2(mean_sin, mean_cos)
    ) + 360
) % 360

print(
    "Circular mean wind direction FROM:",
    round(prevailing_wd, 1),
    "degrees"
)

print(
    "Layout design direction:",
    LAYOUT_WIND_DIRECTION,
    "degrees (South)"
)


# ============================================================
# 7. LOAD POWER AND ESTIMATED CT CURVES
# ============================================================

power_curves = {}
turbine_models = {}

for key, info in TURBINES.items():

    curve = pd.read_csv(CURVE_FILES[key])

    required_curve_columns = [
        "ws_ms",
        "power_kw",
        "ct_estimated"
    ]

    for column in required_curve_columns:

        if column not in curve.columns:

            raise ValueError(
                f"{key} CSV is missing {column}"
            )

    curve = (
        curve.sort_values("ws_ms")
        .drop_duplicates(subset="ws_ms")
    )

    if curve[required_curve_columns].isna().any().any():

        raise ValueError(
            f"{key} CSV contains missing curve values."
        )

    power_curves[key] = curve

    # Create PyWake power and thrust function

    power_ct = PowerCtTabular(
        curve["ws_ms"].to_numpy(dtype=float),
        curve["power_kw"].to_numpy(dtype=float),
        "kW",
        curve["ct_estimated"].to_numpy(dtype=float),
        ws_cutout=info["cut_out"]
    )

    # Create turbine model

    turbine_models[key] = WindTurbine(
        name=info["name"],
        diameter=info["diameter"],
        hub_height=info["hub_height"],
        powerCtFunction=power_ct
    )


# ============================================================
# 8. CREATE WIND SITE
# ============================================================

site = UniformSite(
    p_wd=np.ones(360) / 360,
    ti=TURBULENCE_INTENSITY,
    shear=PowerShear(
        h_ref=REFERENCE_HEIGHT,
        alpha=SHEAR_EXPONENT
    )
)


# ============================================================
# 9. CREATE ALIGNED AND STAGGERED LAYOUTS
#
# Coordinates:
# x = East-West
# y = North-South
#
# At 180 degrees, wind travels north.
# ============================================================

def create_layout(
    n,
    diameter,
    layout_type,
    wind_from_deg=LAYOUT_WIND_DIRECTION
):

    rows = math.ceil(math.sqrt(n))

    columns = math.ceil(n / rows)

    # Fixed crosswind positions.
    # This keeps the aligned rows on common columns.

    cross_grid = (
        np.arange(columns) - (columns - 1) / 2
    ) * CROSSWIND_SPACING * diameter

    # Downwind positions

    down_grid = (
        np.arange(rows) - (rows - 1) / 2
    ) * DOWNWIND_SPACING * diameter

    cross_positions = []
    down_positions = []

    turbine_count = 0

    for row in range(rows):

        remaining = n - turbine_count

        count = min(columns, remaining)

        # Centre incomplete rows on the available grid
        start = (columns - count) // 2

        selected_positions = cross_grid[
            start:start + count
        ]

        # Offset alternate rows for staggered layout

        offset = 0

        if layout_type == "staggered" and row % 2 == 1:

            offset = (
                CROSSWIND_SPACING * diameter / 2
            )

        for cross in selected_positions:

            cross_positions.append(cross + offset)

            down_positions.append(down_grid[row])

            turbine_count += 1

    cross = np.array(cross_positions)

    down = np.array(down_positions)

    # Convert wind FROM direction into downstream direction.
    # Meteorological convention:
    # 180 degrees = wind comes from south and travels north.

    theta = np.radians(wind_from_deg)

    down_x = -np.sin(theta)
    down_y = -np.cos(theta)

    cross_x = np.cos(theta)
    cross_y = -np.sin(theta)

    x = cross * cross_x + down * down_x

    y = cross * cross_y + down * down_y

    return x, y


# ============================================================
# 10. FUNCTION TO DRAW CLEAR TURBINE SYMBOLS
#
# Rotor lines are perpendicular to incoming wind.
# For southerly wind, rotor lines are east-west.
# ============================================================

def draw_turbines(
    ax,
    x,
    y,
    diameter,
    wind_from_deg=LAYOUT_WIND_DIRECTION
):

    theta = np.radians(wind_from_deg)

    # Crosswind unit vector: direction of rotor line

    cross_x = np.cos(theta)
    cross_y = -np.sin(theta)

    for i in range(len(x)):

        # Rotor line endpoints

        half = diameter / 2

        x1 = x[i] - half * cross_x
        x2 = x[i] + half * cross_x

        y1 = y[i] - half * cross_y
        y2 = y[i] + half * cross_y

        # Draw rotor

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="black",
            linewidth=2.5,
            zorder=4
        )

        # Turbine centre

        ax.scatter(
            x[i],
            y[i],
            color="black",
            s=22,
            zorder=5
        )

        # Alternate labels to prevent overlap

        label_y = 12 if i % 2 == 0 else -19

        ax.annotate(
            f"T{i+1}",
            (x[i], y[i]),
            xytext=(7, label_y),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85
            ),
            zorder=6
        )


# ============================================================
# 11. RUN SIX PYWake ANNUAL SIMULATIONS
# ============================================================

results = []

layout_data = {}

for key, info in TURBINES.items():

    turbine = turbine_models[key]

    n = info["number"]

    diameter = info["diameter"]

    installed_capacity = (
        n * info["rated_power"]
    )

    print("\n================================")

    print(info["name"])

    print("Number of turbines:", n)

    print(
        "Installed capacity:",
        installed_capacity,
        "MW"
    )

    print("================================")

    # Store both layouts, oriented for the dominant southerly wind

    for layout_type in ["aligned", "staggered"]:

        x, y = create_layout(
            n,
            diameter,
            layout_type,
            LAYOUT_WIND_DIRECTION
        )

        layout_data[(key, layout_type)] = (x, y)

        # Create wake model

        wake_model = NOJ(site, turbine)

        print(
            f"Running {key} - {layout_type} simulation..."
        )

        # Use actual hourly wind speed and direction.
        # PyWake calculates wake effects for each wind direction.

        simulation = wake_model(
            x,
            y,
            ws=ws,
            wd=wd,
            time=True
        )

        # Total farm power in watts

        farm_power = (
            simulation.Power.sum(dim="wt").values
        )

        # One-hour observations converted to MWh

        net_energy = np.sum(farm_power) / 1e6

        # Annualise valid observations to 8760 hours.
        # Assumes missing hours are representative.

        annual_factor = 8760 / len(ws)

        net_aep = net_energy * annual_factor

        # Gross AEP without wake effects

        ws_hub = ws * (
            info["hub_height"] / REFERENCE_HEIGHT
        ) ** SHEAR_EXPONENT

        curve = power_curves[key]

        gross_power_kw = np.interp(
            ws_hub,
            curve["ws_ms"].to_numpy(),
            curve["power_kw"].to_numpy()
        )

        # Apply the same turbine-specific cut-out used in the PyWake model.
        # Power is zero when hub-height wind speed is above the cut-out speed.
        gross_power_kw[ws_hub > info["cut_out"]] = 0.0

        gross_aep = (
            np.sum(gross_power_kw)
            * n
            / 1000
        ) * annual_factor

        # Wake losses

        wake_loss = gross_aep - net_aep

        wake_loss_percent = (
            wake_loss / gross_aep * 100
            if gross_aep > 0 else np.nan
        )

        # Capacity factor

        capacity_factor = (
            net_aep
            / (installed_capacity * 8760)
            * 100
        )

        # Approximate rectangular layout footprint

        width = np.max(x) - np.min(x) + diameter

        height = np.max(y) - np.min(y) + diameter

        area_hectares = (
            width * height / 10000
        )

        results.append({

            "Turbine": info["name"],

            "Model": key,

            "Number of turbines": n,

            "Layout": layout_type,

            "Installed capacity (MW)": installed_capacity,

            "Layout area (hectares)": area_hectares,

            "Gross AEP (MWh/year)": gross_aep,

            "Net AEP (MWh/year)": net_aep,

            "Wake loss (MWh/year)": wake_loss,

            "Wake loss (%)": wake_loss_percent,

            "Capacity factor (%)": capacity_factor
        })

        print("\nLayout:", layout_type)

        print(
            "Approximate layout area:",
            round(area_hectares, 1),
            "hectares"
        )

        print(
            "Gross AEP:",
            round(gross_aep, 2),
            "MWh/year"
        )

        print(
            "Net AEP:",
            round(net_aep, 2),
            "MWh/year"
        )

        print(
            "Wake loss:",
            round(wake_loss_percent, 2),
            "%"
        )

        print(
            "Capacity factor:",
            round(capacity_factor, 2),
            "%"
        )


# ============================================================
# 12. DISPLAY RESULTS TABLE
# No automatic saving.
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== FINAL RESULTS ==========")

print(
    results_df.round(2).to_string(index=False)
)


# ============================================================
# 13. GRAPH 1 - STAGGERED LAYOUTS
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 7)
)

for ax, (key, info) in zip(
    axes,
    TURBINES.items()
):

    x, y = layout_data[(key, "staggered")]

    draw_turbines(
        ax,
        x,
        y,
        info["diameter"]
    )

    margin = 1.5 * info["diameter"]

    ax.set_xlim(
        np.min(x) - margin,
        np.max(x) + margin
    )

    ax.set_ylim(
        np.min(y) - margin,
        np.max(y) + margin
    )

    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{key} - Staggered\n"
        f"{info['number']} turbines | "
        f"{info['number'] * info['rated_power']} MW"
    )

    ax.set_xlabel("East-West distance (m)")

    ax.set_ylabel("North-South distance (m)")

    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Offshore Wind Farm - Staggered Layouts\n"
    "Designed for dominant southerly wind"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. GRAPH 2 - LINEAR / ALIGNED LAYOUTS
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 7)
)

for ax, (key, info) in zip(
    axes,
    TURBINES.items()
):

    x, y = layout_data[(key, "aligned")]

    draw_turbines(
        ax,
        x,
        y,
        info["diameter"]
    )

    margin = 1.5 * info["diameter"]

    ax.set_xlim(
        np.min(x) - margin,
        np.max(x) + margin
    )

    ax.set_ylim(
        np.min(y) - margin,
        np.max(y) + margin
    )

    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{key} - Aligned\n"
        f"{info['number']} turbines | "
        f"{info['number'] * info['rated_power']} MW"
    )

    ax.set_xlabel("East-West distance (m)")

    ax.set_ylabel("North-South distance (m)")

    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Offshore Wind Farm - Linear / Aligned Layouts\n"
    "Designed for dominant southerly wind"
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. GRAPH 3 - AEP BAR CHART
# STAGGERED VS ALIGNED
# ============================================================

models = ["V164", "SG10", "V236"]

staggered_aep = []

aligned_aep = []

for key in models:

    subset = results_df[
        results_df["Model"] == key
    ]

    staggered_value = subset[
        subset["Layout"] == "staggered"
    ]["Net AEP (MWh/year)"].iloc[0] / 1000

    aligned_value = subset[
        subset["Layout"] == "aligned"
    ]["Net AEP (MWh/year)"].iloc[0] / 1000

    staggered_aep.append(staggered_value)

    aligned_aep.append(aligned_value)


plt.figure(figsize=(11, 7))

x = np.arange(len(models))

width = 0.35

bars1 = plt.bar(
    x - width / 2,
    staggered_aep,
    width,
    label="Staggered"
)

bars2 = plt.bar(
    x + width / 2,
    aligned_aep,
    width,
    label="Linear / Aligned"
)

# Values above bars

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

plt.xlabel("Wind Turbine Model")

plt.ylabel(
    "Net Annual Energy Production (GWh/year)"
)

plt.title(
    "Net Annual Energy Production:\n"
    "Staggered vs Linear / Aligned Layouts"
)

plt.xticks(x, models)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. GRAPH 4 - POWER CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for key, info in TURBINES.items():

    curve = power_curves[key]

    plt.plot(
        curve["ws_ms"],
        curve["power_kw"] / 1000,
        linewidth=2,
        label=info["name"]
    )

plt.xlabel("Wind speed (m/s)")

plt.ylabel("Power output (MW)")

plt.title(
    "Power Curve Comparison - Three Turbine Models"
)

plt.xlim(0, 25)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 17. GRAPH 5 - ESTIMATED CT CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for key, info in TURBINES.items():

    curve = power_curves[key]

    plt.plot(
        curve["ws_ms"],
        curve["ct_estimated"],
        marker="o",
        markersize=3,
        linewidth=2,
        label=info["name"]
    )

plt.xlabel("Wind speed (m/s)")

plt.ylabel("Thrust coefficient (Ct)")

plt.title(
    "Estimated Thrust Coefficient (Ct) Curves"
)

plt.xlim(0, 25)

plt.ylim(bottom=0)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 18. SIX WAKE MAPS
#
# All maps use a southerly wind (180 degrees).
# Wind travels from the bottom of the map towards the top.
#
# The map is north-up, south-down.
# ============================================================

print("\n========== GENERATING WAKE MAPS ==========")

for key, info in TURBINES.items():

    turbine = turbine_models[key]

    n = info["number"]

    diameter = info["diameter"]

    for layout_type in ["aligned", "staggered"]:

        print(
            f"Generating {key} - {layout_type} wake map..."
        )

        # Create layout for southerly wind

        x, y = create_layout(
            n,
            diameter,
            layout_type,
            WAKE_WIND_DIRECTION
        )

        # Create wake model

        wake_model = NOJ(site, turbine)

        # Simulate one selected wind condition

        simulation = wake_model(
            x,
            y,
            ws=[WAKE_WIND_SPEED],
            wd=[WAKE_WIND_DIRECTION]
        )

        # Define grid around the farm

        margin = 5 * diameter

        grid = HorizontalGrid(
            x=np.linspace(
                np.min(x) - margin,
                np.max(x) + margin,
                WAKE_GRID_POINTS
            ),
            y=np.linspace(
                np.min(y) - margin,
                np.max(y) + margin,
                WAKE_GRID_POINTS
            )
        )

        # Calculate wake field

        flow_map = simulation.flow_map(
            grid=grid,
            wd=WAKE_WIND_DIRECTION,
            ws=WAKE_WIND_SPEED
        )

        # Plot wake map without default turbine symbols

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        flow_map.plot_wake_map(
            plot_windturbines=False,
            levels=30,
            ax=ax
        )

        # Draw clear rotor lines and labels

        draw_turbines(
            ax,
            x,
            y,
            diameter,
            WAKE_WIND_DIRECTION
        )

        ax.set_title(
            f"{info['name']} - {layout_type.title()} Layout\n"
            f"Wake map: Southerly wind "
            f"({WAKE_WIND_DIRECTION} degrees), "
            f"{WAKE_WIND_SPEED} m/s"
        )

        ax.set_xlabel("East-West distance (m)")

        ax.set_ylabel("North-South distance (m)")

        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        plt.show()


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n==========================================")

print("OFFSHORE WIND FARM SIMULATION COMPLETED")

print("==========================================")

print("\nSix scenarios completed:")

print("- V164 aligned")

print("- V164 staggered")

print("- SG10 aligned")

print("- SG10 staggered")

print("- V236 aligned")

print("- V236 staggered")

print("\nGraphs displayed:")

print("- Staggered turbine layouts")

print("- Linear / aligned turbine layouts")

print("- AEP comparison bar chart")

print("- Power curve comparison")

print("- Estimated CT curve comparison")

print("- Six wake maps")

print("\nNo graphs or files were automatically saved.")

print("\nRemember: these are preliminary offshore estimates.")
