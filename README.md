## installing YAIB dependencies:

### 1.) Clone YAIB within a location close to but not in this repo

(powershell)
```powershell
git clone https://github.com/rvandewater/YAIB.git
```

sanity check:
```powershell
cd YAIB
ls
```

(you should see a long list of files)

---

### 2.) Create venv:

```powershell
python -m venv venv
```

enter venv:
```powershell
.\venv\Scripts\activate
```

---

### 3.) install dependencys

```powershell
python -m pip install --upgrade pip setuptools wheel
```

(this could take a sec)

## Run Baseline:

```powershell
icu-benchmarks train -d demo_data/mortality24/mimic_demo -n mimic_demo -t BinaryClassification -tn Mortality24 -m LSTM -gc -lc -s 2222 -l ../yaib_logs/ --no-tune -hp train_common.num_workers=0
```
-NOTE: by editing the num_workers variable in the command causes a proper basline output with a "SHAP" value error.

## Tests
Used for running baselines and experiments