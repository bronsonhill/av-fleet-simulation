"""to help understand how differnt param values affect things, not super important for actual repoirt.
to run different plots, commnet/uncomment corresponding labelled lines;
1st run = sensitivity
2nd run = normalised 
python3 -m uv run python plot_lc_calibration.py
"""
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv("lc_calibration_results.csv")           # -> sensitivity
df = pd.read_csv("lc_calibration_normalised_results.csv") # -> normalised

# ------------------ HEATMAP ------------------ #
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True) 
for ax, density in zip(axes, sorted(df["density"].unique())):
    subset = df[df["density"] == density]
    #table = subset.pivot_table( index="gain", columns="cooldown", values="lane_changes", aggfunc="mean") # -> sensitivity
    table = subset.pivot_table( index="gain", columns="cooldown", values="lc_per_veh_km", aggfunc="mean") # -> normalised
    vmin =0                          # -> normalised 
    vmax = df["lc_per_veh_km"].max() # -> normalised 
    #heatmap = ax.imshow(table, cmap="viridis", aspect="auto")                      # -> sensitivity
    heatmap = ax.imshow(table, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax) # -> normalised
    ax.set_title(f"Density = {density}")
    ax.set_xlabel("Cooldown (steps)")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_ylabel("Minimum gain")
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            #ax.text(j, i, f"{table.iloc[i, j]:.0f}", ha="center", va="center", color="white") # -> sensitivity
            ax.text(j, i, f"{table.iloc[i, j]:.3f}", ha="center", va="center", color="white")  # -> normalised
#fig.colorbar(heatmap, ax=axes, label="Mean lane changes")          # -> sensitivity
fig.colorbar(heatmap, ax=axes, label="Lane changes per vehicle-km") # -> normalised
fig.suptitle("Lane-changing activity by parameter configuration")
#plt.show()

# ------------------ LINE PLOT ------------------ #
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, density in zip(axes, sorted(df["density"].unique())):
    subset = df[(df["density"] == density) & (df["cooldown"] == 3)]
    for seed, group in subset.groupby("seed"):
        group = group.sort_values("gain")
        #ax.plot( group["gain"], group["lane_changes"], marker="o", label=f"Seed {seed}") # -> sensitivity
        ax.plot( group["gain"], group["lc_per_veh_km"], marker="o", label=f"Seed {seed}") # -> normalised
    ax.set_title(f"Density = {density}")
    ax.set_xlabel("Minimum gain")
    ax.set_xticks(sorted(df["gain"].unique()))
    ax.grid(alpha=0.3)
#axes[0].set_ylabel("Total lane changes")         # -> sensitivity
axes[0].set_ylabel("Lane changes per vehicle-km") # -> normalised
axes[-1].legend(title="Random seed")
fig.suptitle("Lane-change variation across random seeds")
plt.show()

"""
SENSITIVITY (1st run of calibrate_lc.py):
- at higher densities (40, 60) LC_MIN_GAIN has a noticeable effect on lc frequency, where increasiung param reduces lc activity 
- at density=20 highest LC_MIN_GAIN value increases lc (model behaviour depends on evolving traffic conditions & rand. init...? rather than just threshold)
- over tested range of 2-5 steps LC_COOLDOWN has little effect/influence on lc count
- varied gain & cd settings generally produced smaller speed differences (as opposed to increasing density which predictlyt slows avg speed)

NORMALISED (2nd run of calibrate_lc.py):
- pretty much same as before wrt effect of both params on lc frequency
- substantial changes in lc frequency has minimal effect on avg speed => maximising lc frequency is a bit pointless
- will keep lc_cooldown as =3 given that changing this number doesn't actually do all that much for the current model (provisional)
- need to find real-world data to select appropriate/realistic lc_min_gain value (for now =1.0 is provisional. even tho supported by Gipps)
"""