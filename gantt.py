import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

# Data
task = [
    "planning", "data cleaning", "EDA", "modeling", "deployment",
    "monitoring", "reporting", "communication", "collaboration", "documentation"
]
startdate = [
    "2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01",
    "2024-03-15", "2024-04-01", "2024-04-15", "2024-05-01", "2024-05-15"
]
enddate = [
    "2024-01-14", "2024-01-31", "2024-02-14", "2024-02-29", "2024-03-14",
    "2024-03-31", "2024-04-14", "2024-04-30", "2024-05-14", "2024-05-31"
]

# DataFrame
df = pd.DataFrame({"task": task, "startdate": startdate, "enddate": enddate})
df["startdate"] = pd.to_datetime(df["startdate"])
df["enddate"] = pd.to_datetime(df["enddate"])
df["days"] = (df["enddate"] - df["startdate"]).dt.days
df["color"] = list(plt.cm.tab10.colors[: len(df)])

print(df)

# Figure
fig = plt.figure(figsize=(12, 8))

# Convert dates to ordinal for bar positions
plt.barh(
    y=df["task"],
    width=df["days"],
    left=df["startdate"].map(lambda d: d.toordinal()),
    color=df["color"]
)

# Axis limits
plt.xlim(date(2024, 1, 1).toordinal(), date(2024, 6, 30).toordinal())
plt.ylim(-1, len(df["task"]))

# X ticks (months)
dt_rng = pd.date_range(start=date(2024, 1, 1), end=date(2024, 6, 30), freq="MS")
plt.yticks(range(len(df["task"])), df["task"])
plt.xticks(
    dt_rng.map(lambda d: d.toordinal()),
    [dt.month_name() for dt in dt_rng],
    fontsize=10,
    rotation=45
)

plt.xlabel("Date")
plt.title("Project Roadmap Gantt Chart")
plt.grid(axis="x")
plt.tight_layout()

# Show in VS Code
plt.show()

# Optional: save high-resolution PNG
fig.savefig("project_gantt.png", dpi=300)







