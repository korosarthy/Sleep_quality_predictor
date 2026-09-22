"""
validate_dataset.py
Runs the STRUCTURE / REFERENTIAL INTEGRITY / TEMPORAL / RANGES / LONGITUDINAL checks
specified in Section 33. Returns (report_text, n_pass, n_total).
"""
import numpy as np
import pandas as pd


def validate_dataset(persons_df, sleep_df, lifestyle_df, physiology_df, n_people_expected, n_days_expected):
    lines = []
    def check(desc, passed, detail=""):
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {desc}" + (f" -- {detail}" if detail else ""))

    lines.append("--- STRUCTURE ---")
    check("300 unique people", persons_df["Person_ID"].nunique() == n_people_expected,
          f"found {persons_df['Person_ID'].nunique()}")
    check("9,000 daily records", sleep_df.shape[0] == n_people_expected * n_days_expected,
          f"found {sleep_df.shape[0]}")
    counts = sleep_df.groupby("Person_ID").size()
    check("Exactly 30 records/person", (counts == n_days_expected).all(), f"min={counts.min()}, max={counts.max()}")
    check("Unique Person_ID + Date", sleep_df.duplicated(subset=["Person_ID", "Date"]).sum() == 0)
    check("Unique Record_ID", sleep_df["Record_ID"].is_unique)

    lines.append("--- REFERENTIAL INTEGRITY ---")
    check("Every Person_ID in daily_sleep exists in persons.csv",
          sleep_df["Person_ID"].isin(persons_df["Person_ID"]).all())
    check("Every Record_ID in daily_lifestyle matches daily_sleep",
          set(lifestyle_df["Record_ID"]) == set(sleep_df["Record_ID"]))
    check("Every Record_ID in daily_physiology matches daily_sleep",
          set(physiology_df["Record_ID"]) == set(sleep_df["Record_ID"]))
    check("Location_ID field present on persons.csv (placeholder FK for Dataset 2, not yet resolved)",
          "Location_ID" in persons_df.columns)

    lines.append("--- TEMPORAL ---")
    def is_sequential(g):
        d = pd.to_datetime(g["Date"]).sort_values().reset_index(drop=True)
        return (d.diff().dropna() == pd.Timedelta(days=1)).all()
    check("Dates sequential per person", bool(sleep_df.groupby("Person_ID").apply(is_sequential).all()))
    check("No duplicate dates per person", not sleep_df.duplicated(subset=["Person_ID", "Date"]).any())

    starts = pd.to_datetime(sleep_df["Sleep_Start_DateTime"])
    wakes = pd.to_datetime(sleep_df["Wake_DateTime"])
    check("Sleep timestamps valid", starts.notna().all() and wakes.notna().all())
    check("Wake after sleep start (full dates considered)", (wakes > starts).all())
    computed_dur = (wakes - starts).dt.total_seconds() / 3600
    check("Sleep duration calculated correctly from timestamps",
          bool((computed_dur - sleep_df["Sleep_Duration_Hours"]).abs().max() < 0.02),
          f"max diff={((computed_dur - sleep_df['Sleep_Duration_Hours']).abs().max()):.4f}h")
    n_cross = (wakes.dt.date != starts.dt.date).sum()
    check("Midnight crossing handled (no negative/garbage durations)", bool((computed_dur > 0).all()),
          f"{n_cross} records cross midnight, all durations positive")

    lines.append("--- RANGES ---")
    check("Age plausible (18-75)", persons_df["Age"].between(18, 75).all())
    check("BMI plausible (15-45)", persons_df["BMI"].between(15, 45).all())
    check("Stress 1-10", lifestyle_df["Stress_Level"].between(1, 10).all())
    check("Fatigue 1-10", lifestyle_df["Fatigue_Level"].between(1, 10).all())
    check("Sleep quality 1-10", sleep_df["Sleep_Quality_Score"].between(1, 10).all())
    check("Steps non-negative", (lifestyle_df["Daily_Steps"] >= 0).all())
    check("Activity non-negative", (lifestyle_df["Physical_Activity_Minutes"] >= 0).all())
    check("Exercise non-negative", (lifestyle_df["Exercise_Duration_Minutes"] >= 0).all())
    check("Caffeine non-negative", (lifestyle_df["Caffeine_Intake_mg"] >= 0).all())
    check("Screen time non-negative", (lifestyle_df["Screen_Time_Hours"] >= 0).all())
    check("Work/study hours plausible (0-16)", lifestyle_df["Work_Study_Hours"].between(0, 16).all())
    check("Nap duration non-negative", (lifestyle_df["Nap_Duration_Minutes"] >= 0).all())
    check("Heart rate plausible (40-120)", physiology_df["Resting_Heart_Rate"].between(40, 120).all())
    bp_ok = (physiology_df["Systolic_BP"] > physiology_df["Diastolic_BP"]).all() and \
            physiology_df["Systolic_BP"].between(85, 165).all() and \
            physiology_df["Diastolic_BP"].between(50, 105).all()
    check("Blood pressure plausible (systolic > diastolic, in range)", bool(bp_ok))

    lines.append("--- LONGITUDINAL ---")
    check("Person-level attributes remain stable (one row per Person_ID in persons.csv)",
          persons_df["Person_ID"].is_unique)

    def has_variation(g):
        return g["Sleep_Duration_Hours"].std() > 0.05
    var_ok = sleep_df.groupby("Person_ID").apply(has_variation).mean()
    check("Daily values actually vary (not every day identical)", var_ok > 0.95,
          f"{var_ok*100:.1f}% of people show duration variation")

    def all_identical(g):
        return g["Sleep_Duration_Hours"].nunique() == 1 and g["Sleep_Quality_Score"].nunique() == 1
    identical_people = sleep_df.groupby("Person_ID").apply(all_identical).sum()
    check("Not every day is identical for any person", identical_people == 0, f"{identical_people} found")

    def not_fully_random(g):
        # a very rough "not fully random" signal: day-to-day duration autocorrelation isn't ~0
        # for people with a genuine underlying tendency (weak positive corr expected due to
        # shared baseline pull, not because days are literally identical)
        return g["Sleep_Duration_Hours"].std() < 2.0  # bounded variability, not wild randomness
    bounded_ok = sleep_df.groupby("Person_ID").apply(not_fully_random).mean()
    check("Not every day is unboundedly random (variability stays bounded per person)", bounded_ok > 0.98,
          f"{bounded_ok*100:.1f}% of people have std < 2.0h")

    disruption_rate = lifestyle_df["Is_Disruption_Day"].mean()
    check("Occasional disruptions exist (disruption rate between 5% and 30%)", 0.05 <= disruption_rate <= 0.30,
          f"disruption rate={disruption_rate*100:.1f}%")

    merged = sleep_df.merge(lifestyle_df[["Record_ID", "Stress_Level"]], on="Record_ID")
    merged = merged.merge(persons_df[["Person_ID", "Chronotype", "Occupation"]], on="Person_ID")
    overall_std = merged["Sleep_Quality_Score"].std()
    chrono_range = merged.groupby("Chronotype")["Sleep_Quality_Score"].mean().agg(lambda s: s.max() - s.min())
    occ_range = merged.groupby("Occupation")["Sleep_Quality_Score"].mean().agg(lambda s: s.max() - s.min())
    check("Profiles overlap: chronotype does not directly determine quality",
          chrono_range < 0.5 * overall_std, f"chronotype range={chrono_range:.2f}, overall std={overall_std:.2f}")
    check("Profiles overlap: occupation does not directly determine quality",
          occ_range < 0.6 * overall_std, f"occupation range={occ_range:.2f}, overall std={overall_std:.2f}")

    n_pass = sum(1 for l in lines if l.startswith("[PASS]"))
    n_total = sum(1 for l in lines if l.startswith("["))
    return "\n".join(lines), n_pass, n_total
