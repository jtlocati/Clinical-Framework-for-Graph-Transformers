#DATA PIPELINE:
"""
Pipeline:
  1. Query BigQuery for cohort (adults, first ICU stay, ≥48 h LOS)
  2. Pull hourly-binned vitals and labs
  3. Merge into a single (stay_id, hour, feature) tensor
  4. Attach in-hospital mortality labels
  5. Cache to disk as .npz so you only query BigQuery once
  """

import os, pickle
import numpy
import pandas
from google.cloud import bigquery 
from sklearn.model_selection import train_test_split
from Task4 import config

#initalize big query:
def get_client() -> bigquery.Client:
    return bigquery.Client(project=config.BQ_Project)

#runs a query and returns a tangable DataFrame result
def BigQuery(SQL: str) -> pandas.DataFrame:
    client = get_client()
    print(f"Query Running....")
    DataFrame = client.query(SQL).to_dataframe()
    print(f"Returned: {len(DataFrame):,} rows, {len(DataFrame.columns)} cols")

#Rob BigQuery (step 1)

COT_SQL = f"""
WITH first_icu AS (
    -- pick each patient's very first ICU stay
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.icu_intime,
        i.icu_outtime,
        i.los_icu,                                       -- days
        a.hospital_expire_flag,                          -- 1 = died in hospital
        ROW_NUMBER() OVER (
            PARTITION BY i.subject_id
            ORDER BY i.icu_intime
        ) AS rn
    FROM `{config.MIMIC}.icustay_detail` i
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
        ON i.hadm_id = a.hadm_id
    WHERE i.admission_age >= {config.MinAge}
)
SELECT
    subject_id,
    hadm_id,
    stay_id,
    icu_intime,
    icu_outtime,
    los_icu,
    hospital_expire_flag AS mortality_label
FROM first_icu
WHERE rn = 1
  AND los_icu >= {config.ObservionWindow / 24}        -- ≥ 48 h LOS
ORDER BY stay_id
"""

def extract_info() -> pandas.DataFrame:
    print("Exstracting MIMIC Data....")
    return bigquery(COT_SQL)

#Bin Hourly Vitals (Step 2)

def SQL_vitals(stay_ids: str) -> str:
    cols = ",\n        ".join(
        f"AVG({f}) AS {f}" for f in config.VitalFeatures
    )
    return f"""
    SELECT
        stay_id,
        TIMESTAMP_DIFF(
            TIMESTAMP_TRUNC(charttime, HOUR),
            icu_intime_trunc,
            HOUR
        ) AS hour,
        {cols}
    FROM (
        SELECT
            v.*,
            TIMESTAMP_TRUNC(i.icu_intime, HOUR) AS icu_intime_trunc
        FROM `{config.MIMIC}.vitalsign` v
        INNER JOIN `{config.MIMIC}.icustay_detail` i
            USING (stay_id)
        WHERE v.stay_id IN ({stay_ids})
    )
    WHERE TIMESTAMP_DIFF(
            TIMESTAMP_TRUNC(charttime, HOUR),
            icu_intime_trunc,
            HOUR
          ) BETWEEN 0 AND {config.ObservionWindow - 1}
    GROUP BY stay_id, hour
    ORDER BY stay_id, hour
    """

def ExtractVitals(stay_ids: numpy.ndarray) -> pandas.DataFrame:
    print("\nExtracting Vitals....")
    csv = ",".join(str(s) for s in stay_ids)
    return BigQuery(SQL_vitals(csv))

#Hourly Binned Labs (Step 3)

def SQL_Labs(stay_ids: str): 
    return f"""
    WITH raw_labs AS (
        -- chemistry
        SELECT
            c.stay_id,
            TIMESTAMP_TRUNC(c.charttime, HOUR) AS chart_hour,
            c.albumin,
            c.aniongap,
            c.bicarbonate,
            c.bilirubin_total,
            c.bun,
            c.calcium,
            c.chloride,
            c.creatinine,
            c.glucose AS glucose_lab,
            c.sodium,
            c.potassium,
            NULL AS hematocrit,
            NULL AS hemoglobin,
            NULL AS platelet,
            NULL AS wbc,
            NULL AS lactate,
            NULL AS ptt,
            NULL AS inr
        FROM `{config.MIMIC}.chemistry` c
        WHERE c.stay_id IN ({stay_ids})

        UNION ALL

        -- complete blood count
        SELECT
            b.stay_id,
            TIMESTAMP_TRUNC(b.charttime, HOUR) AS chart_hour,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            b.hematocrit,
            b.hemoglobin,
            b.platelet,
            b.wbc,
            NULL, NULL, NULL
        FROM `{config.MIMIC}.complete_blood_count` b
        WHERE b.stay_id IN ({stay_ids})

        UNION ALL

        -- coagulation
        SELECT
            g.stay_id,
            TIMESTAMP_TRUNC(g.charttime, HOUR) AS chart_hour,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL,
            NULL,
            g.ptt,
            g.inr
        FROM `{config.MIMIC}.coagulation` g
        WHERE g.stay_id IN ({stay_ids})

        UNION ALL

        -- blood gas (for lactate)
        SELECT
            bg.stay_id,
            TIMESTAMP_TRUNC(bg.charttime, HOUR) AS chart_hour,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL,
            bg.lactate,
            NULL, NULL
        FROM `{config.MIMIC}.bg` bg
        WHERE bg.stay_id IN ({stay_ids})
    ),
    with_hour AS (
        SELECT
            r.*,
            TIMESTAMP_DIFF(
                r.chart_hour,
                TIMESTAMP_TRUNC(i.icu_intime, HOUR),
                HOUR
            ) AS hour
        FROM raw_labs r
        INNER JOIN `{config.MIMIC}.icustay_detail` i
            USING (stay_id)
    )
    SELECT
        stay_id,
        hour,
        AVG(albumin) AS albumin,
        AVG(aniongap) AS aniongap,
        AVG(bicarbonate) AS bicarbonate,
        AVG(bilirubin_total) AS bilirubin_total,
        AVG(bun) AS bun,
        AVG(calcium) AS calcium,
        AVG(chloride) AS chloride,
        AVG(creatinine) AS creatinine,
        AVG(glucose_lab) AS glucose_lab,
        AVG(hematocrit) AS hematocrit,
        AVG(hemoglobin) AS hemoglobin,
        AVG(lactate) AS lactate,
        AVG(platelet) AS platelet,
        AVG(potassium) AS potassium,
        AVG(ptt) AS ptt,
        AVG(inr) AS inr,
        AVG(sodium) AS sodium,
        AVG(wbc) AS wbc
    FROM with_hour
    WHERE hour BETWEEN 0 AND {config.ObservionWindow - 1}
    GROUP BY stay_id, hour
    ORDER BY stay_id, hour
    """

def extract_labs(stay_ids: numpy.ndarray) -> pandas.DataFrame:
    print("Extracting Labs...")
    csv = "'".join(str(s) for s in stay_ids)
    return BigQuery(SQL_Labs(csv))

#Merge (Step 4)

def tensor_build(cohart: pandas.DataFrame, vitals: pandas.DataFrame, labs: pandas.DataFrame):
    #Enter return exsample
    stay_ids = cohart["stay_id"].values
    labels = cohart["mortality_label"].values.astype(numpy.int64)
    N = len(stay_ids)
    T = config.ObservionWindow      #-> 48
    F = config.NumberOfFeatures     #-> 26

    X = numpy.zeros((N, T, F), dtype=numpy.float32)
    M = numpy.zeros((N, T, F), dtype=numpy.float32)

    #pk = stay_id
    grouped_vitals = vitals.groupby("stay_id")
    grouped_labs = labs.groupby("stay_id")

    id_to_idx = {sid: i for i, sid in enumerate(stay_ids)}

    #fill vitals:
    for sid, grp in grouped_vitals:
        if sid not in id_to_idx:
            continue
        i = id_to_idx[sid]
        for _, row in grp.iterrows():
            t = int(row["hour"])
            if 0 <= t < T:
                for j, feat in enumerate(config.VitalFeatures):
                    val = row.get(feat, numpy.nan)
                    if pandas.notna(val):
                        X[i, t, j] = val
                        M[i, t, j] = 1.0

    #Fill labs:
    v_offset = len(config.VitalFeatures)
    for sid, grp in grouped_labs:
        if sid not in id_to_idx:
            continue
        i = id_to_idx[sid]
        for _, row in grp.iterrows():
            t = int(row["hour"])
            if 0 <= t < T:
                for j, feat in enumerate(config.LabFeatures):
                    val = row.get(feat, numpy.nan)
                    if pandas.notna(val):
                        X[i, t, v_offset + j] = val
                        M[i, t, v_offset + j] = 1.0


    # Output build status:
    print("================== Tensor ==================")
    print(f"  X shape: {X.shape}   (patients, hours, features)")
    print(f"  Mortality rate: {labels.mean():.3f}")
    print(f"  Overall missingness: {1 - M.mean():.1%}")
    return X, M, labels, stay_ids


#Normalize, forward fill & split

def forward_fill(X: numpy.ndarray, M: numpy.ndarray) -> numpy.ndarray:
    #Within each patient, forward‑fill observed values along the time axis.

    Xf = X.copy()
    N, T, F = Xf.shape
    for i in range(N):
        for f in range(F):
            last_val = 0.0
            for t in range(T):
                if M[i, t, f] == 1.0:
                    last_val = Xf[i, t, f]
                else:
                    Xf[i, t, f] = last_val
    return Xf

# Find per-feature mean and studio deviation from observed values 
def find_stats(X: numpy.ndarray, M: numpy.ndarray):
    Xflat = X.reshape(-1, X.shape[-1])
    Mflat = M.reshape(-1, M.shape[-1])
    means = numpy.zeros(X.shape[-1], dtype=numpy.float32)
    stds  = numpy.ones(X.shape[-1], dtype=numpy.float32)
    for f in range(X.shape[-1]):
        obs = Xflat[:, f][Mflat[:, f] == 1.0]
        if len(obs) > 1:
            means[f] = obs.mean()
            stds[f]  = obs.std() + 1e-8
    return means, stds

def normalise(X, means, stds):
    return (X - means[None, None, :]) / stds[None, None, :]

#Returns dict with keys 'train', 'val', 'test', each containin (X, M, y, ids) arrays.
def split_data(X, M, y, ids, seed):
    idx = numpy.arange(len(y))
    # first split off test
    idx_train_val, idx_test = train_test_split(
        idx, test_size=config.Test_Frac, random_state=seed, stratify=y
    )
    # then split train/val from remainder
    relative_val = config.ValFrac / (1 - config.Test_Frac)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=relative_val, random_state=seed,
        stratify=y[idx_train_val]
    )
    splits = {}
    for name, ix in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        splits[name] = (X[ix], M[ix], y[ix], ids[ix])
    return splits

#Master pipeline (run once, cache to disk)
CACHE_FILE = os.path.join(config.DATA_DIRECTORY, "mimic4_icu_mortality_48h.npz")


def run_pipeline(force_refresh: bool = False):
    """
    Returns X, M, y, stay_ids  (all np arrays, NOT split yet).
    Splits happen per seed inside the experiment loop.
    """
    if os.path.exists(CACHE_FILE) and not force_refresh:
        print(f"Loading cached data from {CACHE_FILE}")
        data = numpy.load(CACHE_FILE)
        return data["X"], data["M"], data["y"], data["ids"]

    # 1. cohort
    cohort = extract_info()
    stay_ids = cohort["stay_id"].values
    print(f"  Cohort size: {len(cohort):,}")

    # 2. vitals + labs  (chunked if cohort is very large)
    CHUNK = 5000
    vitals_dfs, labs_dfs = [], []
    for start in range(0, len(stay_ids), CHUNK):
        chunk_ids = stay_ids[start : start + CHUNK]
        vitals_dfs.append(ExtractVitals(chunk_ids))
        labs_dfs.append(extract_labs(chunk_ids))
    vitals = pandas.concat(vitals_dfs, ignore_index=True)
    labs   = pandas.concat(labs_dfs, ignore_index=True)

    # 3. build tensor
    X, M, y, ids = build_tensor(cohort, vitals, labs)

    # 4. cache
    np.savez_compressed(CACHE_FILE, X=X, M=M, y=y, ids=ids)
    print(f"  Cached to {CACHE_FILE}")
    return X, M, y, ids


def prepare_for_seed(X_raw, M, y, ids, seed):
    """
    Given raw data, do:
      1. split
      2. compute stats on TRAIN only
      3. forward-fill
      4. normalise
    Returns dict of splits with processed arrays.
    """
    splits = split_data(X_raw, M, y, ids, seed)

    # stats from train set observed values only
    X_tr, M_tr, _, _ = splits["train"]
    means, stds = find_stats(X_tr, M_tr)

    processed = {}
    for name in ["train", "val", "test"]:
        Xi, Mi, yi, idi = splits[name]
        Xi_ff = forward_fill(Xi, Mi)
        Xi_norm = normalise(Xi_ff, means, stds)
        processed[name] = {
            "X": Xi_norm,       # (n, 48, 26) normalised + forward‑filled
            "M": Mi,            # (n, 48, 26) original mask
            "y": yi,            # (n,)
            "ids": idi,
            "X_raw": Xi,        # un‑normalised (for flat models)
        }

    print(f"\n  Seed {seed} splits — "
          f"train {len(processed['train']['y'])} | "
          f"val {len(processed['val']['y'])} | "
          f"test {len(processed['test']['y'])}")
    return processed, means, stds


# quick sanity check if run directly
if __name__ == "__main__":
    X, M, y, ids = run_pipeline()
    print(f"\nDataset: {X.shape[0]} patients, {config.ObservionWindow}h, {config.NumberOfFeatures} features")
    print(f"Mortality rate: {y.mean():.3f}")
    print(f"Missingness: {1 - M.mean():.1%}")