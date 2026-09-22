"""
generate_eda.py
generate_summary_statistics() and generate_visualizations().
All numbers/charts come from the actual generated data.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def generate_summary_statistics(integrated_df, out_csv_path):
    numeric_cols = ["Age", "BMI", "Sleep_Duration_Hours", "Sleep_Quality_Score", "Stress_Level",
                     "Fatigue_Level", "Physical_Activity_Minutes", "Daily_Steps", "Screen_Time_Hours",
                     "Caffeine_Intake_mg", "Resting_Heart_Rate", "Bedtime_Hour", "Wake_Time_Hour",
                     "Sleep_Efficiency", "Sleep_Onset_Latency_Minutes"]
    rows = []
    for col in numeric_cols:
        s = integrated_df[col]
        rows.append({"Variable": col, "Count": int(s.count()), "Mean": round(float(s.mean()), 2),
                     "Median": round(float(s.median()), 2), "Std": round(float(s.std()), 2),
                     "Min": round(float(s.min()), 2), "Max": round(float(s.max()), 2)})
    num_df = pd.DataFrame(rows)

    cat_rows = []
    person_level = integrated_df.drop_duplicates("Person_ID")
    for col in ["Gender", "Occupation", "Chronotype", "BMI_Category"]:
        vc = person_level[col].value_counts()
        for k, v in vc.items():
            cat_rows.append({"Variable": col, "Category": k, "Count": int(v),
                              "Percent": round(100 * v / len(person_level), 2)})
    for col in ["Sleep_Quality_Category"]:
        vc = integrated_df[col].value_counts()
        for k, v in vc.items():
            cat_rows.append({"Variable": col, "Category": k, "Count": int(v),
                              "Percent": round(100 * v / len(integrated_df), 2)})
    cat_df = pd.DataFrame(cat_rows)

    combined = pd.concat([num_df, cat_df], axis=0, ignore_index=True, sort=False)
    combined.to_csv(out_csv_path, index=False)
    return num_df, cat_df


def generate_visualizations(integrated_df, figures_dir, example_person_ids, kaggle_df=None):
    df = integrated_df
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{figures_dir}/{name}.png")
        plt.close(fig)

    # 1. Sleep duration distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Sleep_Duration_Hours"], bins=30, color="#3b6fa0", edgecolor="white")
    ax.set_xlabel("Sleep Duration (hours)"); ax.set_ylabel("Count"); ax.set_title("1. Sleep Duration Distribution")
    save(fig, "sleep_duration_distribution")

    # 2. Sleep quality distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Sleep_Quality_Score"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#3b8f6f")
    ax.set_xlabel("Sleep Quality Score (1-10)"); ax.set_ylabel("Count"); ax.set_title("2. Sleep Quality Distribution")
    save(fig, "sleep_quality_distribution")

    # 3. Stress distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Stress_Level"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#a0653b")
    ax.set_xlabel("Stress Level (1-10)"); ax.set_ylabel("Count"); ax.set_title("3. Stress Distribution")
    save(fig, "stress_distribution")

    # 4. Physical activity distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Physical_Activity_Minutes"], bins=30, color="#6f3ba0", edgecolor="white")
    ax.set_xlabel("Physical Activity (minutes)"); ax.set_ylabel("Count")
    ax.set_title("4. Physical Activity Distribution")
    save(fig, "physical_activity_distribution")

    # 5. Bedtime distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Bedtime_Hour"], bins=40, color="#3b6fa0", edgecolor="white")
    ax.set_xlabel("Bedtime (decimal hour)"); ax.set_ylabel("Count"); ax.set_title("5. Bedtime Distribution")
    save(fig, "bedtime_distribution")

    # 6. Sleep quality vs duration
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Sleep_Duration_Hours"], df["Sleep_Quality_Score"], s=4, alpha=0.15, color="#3b6fa0")
    means = df.groupby(df["Sleep_Duration_Hours"].round())["Sleep_Quality_Score"].mean()
    ax.plot(means.index, means.values, color="black", linewidth=2, label="mean by rounded duration")
    ax.set_xlabel("Sleep Duration (hours)"); ax.set_ylabel("Sleep Quality Score")
    ax.set_title("6. Sleep Quality vs Sleep Duration"); ax.legend()
    save(fig, "sleep_quality_vs_duration")

    # 7. Sleep quality vs stress
    fig, ax = plt.subplots(figsize=(6, 4))
    box = [df.loc[df.Stress_Level == s, "Sleep_Quality_Score"] for s in range(1, 11)]
    ax.boxplot(box, positions=range(1, 11), widths=0.6)
    ax.set_xlabel("Stress Level"); ax.set_ylabel("Sleep Quality Score"); ax.set_title("7. Sleep Quality vs Stress")
    save(fig, "sleep_quality_vs_stress")

    # 8. Sleep quality vs physical activity
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Physical_Activity_Minutes"], df["Sleep_Quality_Score"], s=4, alpha=0.15, color="#6f3ba0")
    bins = pd.cut(df["Physical_Activity_Minutes"], bins=15)
    means2 = df.groupby(bins)["Sleep_Quality_Score"].mean()
    mids = [iv.mid for iv in means2.index]
    ax.plot(mids, means2.values, color="black", linewidth=2, label="mean by activity bin")
    ax.set_xlabel("Physical Activity (minutes)"); ax.set_ylabel("Sleep Quality Score")
    ax.set_title("8. Sleep Quality vs Physical Activity"); ax.legend()
    save(fig, "sleep_quality_vs_activity")

    # 9. 30-day sleep trajectory
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pid in example_person_ids:
        sub = df[df.Person_ID == pid].sort_values("Date")
        ax.plot(range(1, len(sub) + 1), sub["Sleep_Duration_Hours"], marker="o", markersize=3, label=pid)
    ax.set_xlabel("Day"); ax.set_ylabel("Sleep Duration (hours)"); ax.set_title("9. Individual 30-Day Sleep Trajectory")
    ax.legend(fontsize=8)
    save(fig, "individual_sleep_trajectory")

    # 10. Bedtime variability
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df.Person_ID == pid, "Bedtime_Hour"] for pid in example_person_ids]
    ax.boxplot(data, tick_labels=example_person_ids)
    ax.set_ylabel("Bedtime (decimal hour)"); ax.set_title("10. Bedtime Variability")
    save(fig, "bedtime_variability")

    # 11. Sleep quality by chronotype
    fig, ax = plt.subplots(figsize=(6, 4))
    chronos = ["Morning", "Intermediate", "Evening"]
    box = [df.loc[df.Chronotype == c, "Sleep_Quality_Score"] for c in chronos]
    ax.boxplot(box, tick_labels=chronos)
    ax.set_ylabel("Sleep Quality Score"); ax.set_title("11. Sleep Quality by Chronotype")
    save(fig, "sleep_quality_by_chronotype")

    # 12. Sleep quality by occupation
    fig, ax = plt.subplots(figsize=(9, 4.5))
    occs = sorted(df["Occupation"].unique())
    box = [df.loc[df.Occupation == o, "Sleep_Quality_Score"] for o in occs]
    ax.boxplot(box, tick_labels=occs)
    ax.set_ylabel("Sleep Quality Score"); ax.set_title("12. Sleep Quality by Occupation")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "sleep_quality_by_occupation")

    # 13. Weekday vs weekend sleep duration
    fig, ax = plt.subplots(figsize=(5, 4))
    box = [df.loc[~df.Is_Weekend, "Sleep_Duration_Hours"], df.loc[df.Is_Weekend, "Sleep_Duration_Hours"]]
    ax.boxplot(box, tick_labels=["Weekday", "Weekend"])
    ax.set_ylabel("Sleep Duration (hours)"); ax.set_title("13. Weekday vs Weekend Sleep Duration")
    save(fig, "weekday_weekend_comparison")

    # 14. Daytime vs nighttime sleep
    fig, ax = plt.subplots(figsize=(5, 4))
    box = [df.loc[~df.Daytime_Sleep_Flag, "Sleep_Quality_Score"], df.loc[df.Daytime_Sleep_Flag, "Sleep_Quality_Score"]]
    ax.boxplot(box, tick_labels=["Nighttime Sleep", "Daytime Sleep"])
    ax.set_ylabel("Sleep Quality Score"); ax.set_title("14. Daytime vs Nighttime Sleep (Quality)")
    save(fig, "daytime_nighttime_sleep")

    # 15. Correlation matrix
    corr_cols = ["Sleep_Duration_Hours", "Sleep_Quality_Score", "Stress_Level", "Physical_Activity_Minutes",
                 "Caffeine_Intake_mg", "Screen_Time_Hours", "Sleep_Efficiency", "Resting_Heart_Rate",
                 "Fatigue_Level", "Sleep_Onset_Latency_Minutes"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("15. Correlation Matrix")
    save(fig, "correlation_matrix")

    # 16. Synthetic vs Kaggle (only if provided)
    k_status = "skipped - Sleep_health_and_lifestyle_dataset.csv not provided"
    if kaggle_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df["Sleep_Duration_Hours"], bins=25, alpha=0.6, label="Synthetic", color="#3b6fa0", density=True)
        if "Sleep Duration" in kaggle_df.columns:
            axes[0].hist(kaggle_df["Sleep Duration"], bins=25, alpha=0.6, label="Kaggle (real reference)",
                         color="#a0653b", density=True)
        axes[0].set_title("Sleep Duration: Synthetic vs Kaggle"); axes[0].legend()
        axes[1].hist(df["Sleep_Quality_Score"], bins=10, alpha=0.6, label="Synthetic", color="#3b6fa0", density=True)
        if "Quality of Sleep" in kaggle_df.columns:
            axes[1].hist(kaggle_df["Quality of Sleep"], bins=10, alpha=0.6, label="Kaggle (real reference)",
                         color="#a0653b", density=True)
        axes[1].set_title("Sleep Quality: Synthetic vs Kaggle"); axes[1].legend()
        save(fig, "synthetic_vs_kaggle")
        k_status = "generated"

    return k_status
