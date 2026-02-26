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

## Tests
Used for running baselines and experiments