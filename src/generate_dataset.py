"""
generate_dataset.py
Modular synthetic longitudinal data generator for Dataset 1.

Generation order (per spec Section 31):
  generate_persons() -> generate_daily_dates() -> [per person, per day:]
  sleep timing -> generate_lifestyle() -> generate_physiology() -> generate_sleep_quality()
  -> export_datasets()

Sleep_Quality_Score is computed LAST for each day, after that day's lifestyle and
physiology values already exist, since the target depends on them. It is a noisy,
probabilistic function of several factors - never a closed-form deterministic formula.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OCCUPATIONS = [
    "Student", "Office Worker", "Healthcare Worker", "Teacher", "Retail Worker",
    "Software/IT Worker", "Freelancer", "Driver", "Manual Worker",
    "Night-Shift Worker", "Researcher", "Self-Employed",
]
OCC_WEIGHTS = np.array([0.14, 0.15, 0.09, 0.07, 0.08, 0.11, 0.08, 0.05, 0.06, 0.06, 0.05, 0.06])
OCC_WEIGHTS = OCC_WEIGHTS / OCC_WEIGHTS.sum()
GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]
CHRONOTYPES = ["Morning", "Intermediate", "Evening"]
CHRONOTYPE_BIAS = {
    "Student": [0.15, 0.40, 0.45], "Office Worker": [0.30, 0.45, 0.25],
    "Healthcare Worker": [0.25, 0.40, 0.35], "Teacher": [0.35, 0.45, 0.20],
    "Retail Worker": [0.25, 0.45, 0.30], "Software/IT Worker": [0.20, 0.40, 0.40],
    "Freelancer": [0.20, 0.40, 0.40], "Driver": [0.30, 0.40, 0.30],
    "Manual Worker": [0.40, 0.40, 0.20], "Night-Shift Worker": [0.10, 0.30, 0.60],
    "Researcher": [0.20, 0.40, 0.40], "Self-Employed": [0.25, 0.40, 0.35],
}


def _bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25: return "Normal"
    if bmi < 30: return "Overweight"
    return "Obese"


def generate_persons(n_people, rng):
    """Returns (persons_df [public columns only], latent list [internal generation
    parameters used to drive realistic daily records - never exported])."""
    rows, latent = [], []
    for i in range(1, n_people + 1):
        pid = f"P{i:04d}"
        occupation = rng.choice(OCCUPATIONS, p=OCC_WEIGHTS)
        age = int(np.clip(rng.normal(21, 2.5), 18, 30)) if occupation == "Student" \
            else int(np.clip(rng.normal(38, 11), 18, 75))
        gender = rng.choice(GENDERS, p=GENDER_WEIGHTS)
        bmi = float(np.clip(rng.normal(24.5, 4.2), 16.5, 42.0))
        chronotype = rng.choice(CHRONOTYPES, p=CHRONOTYPE_BIAS[occupation])

        is_night_shift = occupation == "Night-Shift Worker"
        chrono_shift = {"Morning": -0.6, "Intermediate": 0.0, "Evening": 0.8}[chronotype]
        if is_night_shift:
            base_bedtime = float(np.clip(rng.normal(8.5, 1.2), 5.0, 12.0))
        else:
            base_bedtime = float(np.clip(23.0 + chrono_shift + rng.normal(0, 0.7), 20.5, 26.5))

        base_duration = float(np.clip(rng.normal(7.3, 0.7), 5.2, 9.2))
        bedtime_sd = float(np.clip(rng.normal(0.55, 0.20), 0.15, 1.4))
        duration_sd = float(np.clip(rng.normal(0.55, 0.15), 0.2, 1.1))
        stress_baseline = float(np.clip(rng.normal(5.0, 1.4), 1.5, 9.0))
        if occupation in ("Healthcare Worker", "Driver", "Manual Worker"):
            stress_baseline = float(np.clip(stress_baseline + rng.normal(0.6, 0.4), 1.5, 9.5))
        activity_baseline = float(np.clip(rng.normal(45, 25), 0, 140))
        if occupation in ("Manual Worker", "Driver"):
            activity_baseline = float(np.clip(activity_baseline + rng.normal(15, 10), 0, 160))
        caffeine_baseline = float(np.clip(rng.normal(140, 90), 0, 450))
        screen_baseline = float(np.clip(rng.normal(5.0, 1.8), 0.5, 11.0))
        if occupation == "Student":
            screen_baseline = float(np.clip(screen_baseline + rng.normal(1.0, 0.6), 0.5, 13.0))
        work_study_baseline = {
            "Student": 5.5, "Office Worker": 8.0, "Healthcare Worker": 8.5, "Teacher": 7.0,
            "Retail Worker": 7.5, "Software/IT Worker": 8.0, "Freelancer": 6.0, "Driver": 8.5,
            "Night-Shift Worker": 8.0, "Manual Worker": 8.0, "Researcher": 7.5, "Self-Employed": 6.5,
        }[occupation]
        work_study_baseline = float(np.clip(work_study_baseline + rng.normal(0, 1.0), 0, 12))
        rhr_baseline = float(np.clip(rng.normal(68 - 0.05 * max(0, activity_baseline - 30), 7), 48, 95))
        sbp_baseline = float(np.clip(rng.normal(118 + 0.15 * max(0, bmi - 25), 8), 95, 148))
        dbp_baseline = float(np.clip(sbp_baseline - rng.normal(38, 4), 58, 96))
        resilience = float(np.clip(rng.normal(0, 1), -2.2, 2.2))
        nap_propensity = 0.35 if occupation in ("Student", "Night-Shift Worker", "Freelancer") else 0.12
        irregular_bonus = 0.25 if occupation in ("Freelancer", "Self-Employed", "Student") else 0.0

        rows.append({
            "Person_ID": pid, "Age": age, "Gender": gender, "Occupation": occupation,
            "BMI": round(bmi, 1), "BMI_Category": _bmi_category(bmi), "Chronotype": chronotype,
            "Location_ID": "",  # placeholder FK for Dataset 2 - not generated this round
        })
        latent.append({
            "Person_ID": pid, "is_night_shift": is_night_shift, "base_bedtime": base_bedtime,
            "base_duration": base_duration, "bedtime_sd": bedtime_sd, "duration_sd": duration_sd,
            "stress_baseline": stress_baseline, "activity_baseline": activity_baseline,
            "caffeine_baseline": caffeine_baseline, "screen_baseline": screen_baseline,
            "work_study_baseline": work_study_baseline, "rhr_baseline": rhr_baseline,
            "sbp_baseline": sbp_baseline, "dbp_baseline": dbp_baseline, "resilience": resilience,
            "nap_propensity": nap_propensity, "irregular_bonus": irregular_bonus,
        })
    return pd.DataFrame(rows), latent


def generate_daily_dates(start_date, n_days):
    """N_DAYS consecutive calendar dates, shared across all people."""
    return [start_date + timedelta(days=d) for d in range(n_days)]


def _generate_sleep_timing(L, date, is_weekend, rng):
    """Generates Sleep_Start_DateTime / Wake_DateTime and everything derivable from them."""
    is_disruption = rng.random() < (0.12 + L["irregular_bonus"] * 0.4)
    disruption_type = rng.choice(
        ["late_bedtime", "short_sleep", "high_stress", "high_screen", "high_caffeine", "low_activity"]
    ) if is_disruption else None

    we_shift = rng.normal(0.8, 0.5) if (is_weekend and not L["is_night_shift"]) else 0.0
    disruption_bedtime_shift = rng.uniform(1.5, 3.5) if disruption_type == "late_bedtime" else 0.0
    disruption_duration_shift = -rng.uniform(1.5, 3.2) if disruption_type == "short_sleep" else 0.0

    bedtime = L["base_bedtime"] + we_shift + disruption_bedtime_shift + rng.normal(0, L["bedtime_sd"])
    duration = L["base_duration"] + disruption_duration_shift + rng.normal(0, L["duration_sd"])
    duration = float(np.clip(duration, 2.8, 11.5))
    bedtime_hour_24 = bedtime % 24

    sleep_start_dt = datetime(date.year, date.month, date.day) + timedelta(hours=float(bedtime_hour_24))
    wake_dt = sleep_start_dt + timedelta(hours=duration)
    sleep_start_dt = sleep_start_dt.replace(second=0, microsecond=0)
    wake_dt = wake_dt.replace(second=0, microsecond=0)
    duration = (wake_dt - sleep_start_dt).total_seconds() / 3600.0

    cross_midnight = wake_dt.date() != sleep_start_dt.date()
    midpoint_dt = sleep_start_dt + (wake_dt - sleep_start_dt) / 2
    midpoint_hour = midpoint_dt.hour + midpoint_dt.minute / 60
    daytime_sleep = 9.0 <= midpoint_hour <= 17.0

    return {
        "sleep_start_dt": sleep_start_dt, "wake_dt": wake_dt, "duration": duration,
        "bedtime_hour_24": bedtime_hour_24, "midpoint_hour": midpoint_hour,
        "cross_midnight": cross_midnight, "daytime_sleep": daytime_sleep,
        "is_disruption": is_disruption, "disruption_type": disruption_type,
    }


def generate_lifestyle(L, timing, is_weekend, prev_duration_deficit, rng):
    """Generates one day's lifestyle record, given that day's sleep timing outcome."""
    dt = timing["disruption_type"]
    stress_add = rng.uniform(2.0, 4.0) if dt == "high_stress" else 0.0
    screen_add = rng.uniform(2.0, 4.5) if dt == "high_screen" else 0.0
    caffeine_add = rng.uniform(120, 260) if dt == "high_caffeine" else 0.0
    activity_mult = rng.uniform(0.1, 0.4) if dt == "low_activity" else 1.0
    weekend_work_mult = 0.35 if is_weekend else 1.0

    stress = np.clip(L["stress_baseline"] + stress_add + rng.normal(0, 1.1) + 0.4 * prev_duration_deficit, 1, 10)
    activity = np.clip((L["activity_baseline"] + rng.normal(0, 18)) * activity_mult, 0, 220)
    steps = np.clip(activity * rng.uniform(90, 130) + rng.normal(0, 800), 0, 30000)
    exercise = np.clip(activity * rng.uniform(0.15, 0.45) + rng.normal(0, 5), 0, 150)
    caffeine = np.clip(L["caffeine_baseline"] + caffeine_add + rng.normal(0, 45) + 15 * prev_duration_deficit, 0, 600)
    screen = np.clip(L["screen_baseline"] + screen_add + rng.normal(0, 1.0), 0, 16)
    work_study = np.clip(L["work_study_baseline"] * weekend_work_mult + rng.normal(0, 1.0), 0, 14)
    fatigue = np.clip(4.5 + 0.9 * prev_duration_deficit + 0.3 * (stress - 5) + rng.normal(0, 1.2), 1, 10)

    nap_prob = L["nap_propensity"] + (0.15 if timing["is_disruption"] else 0) + (0.1 if timing["duration"] < 6 else 0)
    nap_occurred = rng.random() < np.clip(nap_prob, 0, 0.8)
    nap_start_time, nap_duration = "", 0
    if nap_occurred:
        nap_hour = rng.uniform(13, 18) if not L["is_night_shift"] else rng.uniform(19, 22)
        nh, nm = int(nap_hour), int((nap_hour - int(nap_hour)) * 60)
        nap_start_time = f"{nh:02d}:{nm:02d}"
        nap_duration = int(np.clip(rng.normal(45, 25), 10, 150))

    return {
        "Physical_Activity_Minutes": round(float(activity), 1), "Daily_Steps": int(steps),
        "Exercise_Duration_Minutes": round(float(exercise), 1), "Caffeine_Intake_mg": round(float(caffeine), 1),
        "Screen_Time_Hours": round(float(screen), 2), "Work_Study_Hours": round(float(work_study), 2),
        "Stress_Level": int(round(stress)), "Fatigue_Level": int(round(fatigue)),
        "Nap_Occurred": nap_occurred, "Nap_Start_Time": nap_start_time, "Nap_Duration_Minutes": nap_duration,
        "Is_Disruption_Day": timing["is_disruption"],
        "_stress_raw": stress, "_activity_raw": activity,  # internal, used by generate_physiology/generate_sleep_quality
    }


def generate_physiology(L, lifestyle, rng):
    """Generates one day's physiological record from person baselines + that day's stress/activity."""
    stress, activity = lifestyle["_stress_raw"], lifestyle["_activity_raw"]
    rhr = np.clip(L["rhr_baseline"] + rng.normal(0, 3.5) - 0.03 * activity, 42, 118)
    sbp = np.clip(L["sbp_baseline"] + rng.normal(0, 4.5) + 0.4 * (stress - 5), 90, 160)
    dbp = np.clip(L["dbp_baseline"] + rng.normal(0, 3.5) + 0.2 * (stress - 5), 55, 100)
    return {"Resting_Heart_Rate": int(round(rhr)), "Systolic_BP": int(round(sbp)), "Diastolic_BP": int(round(dbp))}


def generate_sleep_quality(L, timing, lifestyle, efficiency, rng):
    """Generates Sleep_Quality_Score LAST, as a stochastic (non-deterministic) function of the
    day's already-generated variables. Never a closed-form additive formula - the base score is
    nudged by several soft, overlapping factors and then perturbed with irreducible noise."""
    duration_dev = timing["duration"] - L["base_duration"]
    stress = lifestyle["_stress_raw"]
    activity = lifestyle["_activity_raw"]

    score = 6.6
    score += -0.55 * abs(duration_dev) * (1 if duration_dev < 0 else 0.6)
    score += -0.28 * (stress - 5)
    score += 0.018 * (efficiency - 85)
    score += 0.15 * L["resilience"]
    score += 0.01 * (activity - 45) if activity <= 100 else -0.01 * (activity - 100)
    score += -0.35 if timing["is_disruption"] else 0.05
    score += rng.normal(0, 1.15)  # irreducible noise - keeps the target non-deterministic

    quality = int(np.clip(round(score), 1, 10))
    category = "Good" if quality >= 7 else ("Average" if quality >= 4 else "Poor")
    return quality, category


def generate_all_daily_records(persons_df, latent_list, dates, rng):
    """Orchestrates the per-person, per-day generation in the required order:
    timing -> lifestyle -> physiology -> quality (last)."""
    sleep_rows, lifestyle_rows, physiology_rows = [], [], []
    rec_counter = 1
    for L in latent_list:
        pid = L["Person_ID"]
        prev_duration_deficit = 0.0
        for date in dates:
            dow = date.strftime("%A")
            is_weekend = dow in ("Saturday", "Sunday")

            timing = _generate_sleep_timing(L, date, is_weekend, rng)
            lifestyle = generate_lifestyle(L, timing, is_weekend, prev_duration_deficit, rng)
            physiology = generate_physiology(L, lifestyle, rng)

            onset_latency = int(np.clip(rng.normal(15 + 2 * max(0, lifestyle["_stress_raw"] - 5), 8), 0, 90))
            awakenings = int(np.clip(rng.poisson(1.2 + 0.5 * max(0, lifestyle["_stress_raw"] - 5) / 5), 0, 8))
            efficiency = np.clip(94 - 2.2 * awakenings - 1.5 * max(0, 7.3 - timing["duration"])
                                  + rng.normal(0, 3.5), 55, 100)

            quality, category = generate_sleep_quality(L, timing, lifestyle, efficiency, rng)

            prev_duration_deficit = max(0.0, L["base_duration"] - timing["duration"])
            rid = f"R{rec_counter:06d}"
            rec_counter += 1

            sleep_rows.append({
                "Record_ID": rid, "Person_ID": pid, "Date": date.strftime("%Y-%m-%d"),
                "Day_of_Week": dow, "Is_Weekend": is_weekend,
                "Sleep_Start_DateTime": timing["sleep_start_dt"].strftime("%Y-%m-%d %H:%M"),
                "Wake_DateTime": timing["wake_dt"].strftime("%Y-%m-%d %H:%M"),
                "Sleep_Onset_Latency_Minutes": onset_latency, "Number_of_Awakenings": awakenings,
                "Sleep_Efficiency": round(float(efficiency), 1),
                "Sleep_Duration_Hours": round(timing["duration"], 2),
                "Bedtime_Hour": round(timing["bedtime_hour_24"], 2),
                "Wake_Time_Hour": round(timing["wake_dt"].hour + timing["wake_dt"].minute / 60, 2),
                "Sleep_Midpoint_Hour": round(timing["midpoint_hour"], 2),
                "Cross_Midnight_Flag": timing["cross_midnight"], "Daytime_Sleep_Flag": timing["daytime_sleep"],
                "Sleep_Quality_Score": quality, "Sleep_Quality_Category": category,
            })
            lifestyle_rows.append({
                "Record_ID": rid, "Person_ID": pid,
                **{k: v for k, v in lifestyle.items() if not k.startswith("_")},
            })
            physiology_rows.append({"Record_ID": rid, "Person_ID": pid, **physiology})

    return pd.DataFrame(sleep_rows), pd.DataFrame(lifestyle_rows), pd.DataFrame(physiology_rows)


def build_integrated(persons_df, sleep_df, lifestyle_df, physiology_df):
    df = sleep_df.merge(lifestyle_df, on=["Record_ID", "Person_ID"], how="left")
    df = df.merge(physiology_df, on=["Record_ID", "Person_ID"], how="left")
    df = df.merge(persons_df, on="Person_ID", how="left")
    return df


def export_datasets(persons_df, sleep_df, lifestyle_df, physiology_df, integrated_df, out_dirs):
    persons_df.to_csv(f"{out_dirs['synthetic']}/persons.csv", index=False)
    sleep_df.to_csv(f"{out_dirs['synthetic']}/daily_sleep.csv", index=False)
    lifestyle_df.to_csv(f"{out_dirs['synthetic']}/daily_lifestyle.csv", index=False)
    physiology_df.to_csv(f"{out_dirs['synthetic']}/daily_physiology.csv", index=False)
    integrated_df.to_csv(f"{out_dirs['processed']}/integrated_person_day.csv", index=False)
