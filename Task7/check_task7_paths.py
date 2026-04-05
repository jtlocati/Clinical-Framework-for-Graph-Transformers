from pathlib import Path
import re

ROOT = Path(".").resolve()

MOVED = {
    "age_group_counts.csv": "Task7/results/counts/age_group_counts.csv",
    "race_counts.csv": "Task7/results/counts/race_counts.csv",
    "race_intervention_weights.csv": "Task7/results/counts/race_intervention_weights.csv",
    "sex_counts.csv": "Task7/results/counts/sex_counts.csv",
    "task7_cross_seed_summary.csv": "Task7/results/cross_seed_summaries/task7_cross_seed_summary.csv",
    "TGTFull_race_weighted_cap10_vs_original_disparity_summary.csv": "Task7/results/cross_seed_summaries/TGTFull_race_weighted_cap10_vs_original_disparity_summary.csv",
    "TGTFull_race_weighted_cap10_vs_original_overall_summary.csv": "Task7/results/cross_seed_summaries/TGTFull_race_weighted_cap10_vs_original_overall_summary.csv",
    "task7_intervention_comparison.csv": "Task7/results/final_tables/task7_intervention_comparison.csv",
    "task7_intervention_summary.txt": "Task7/results/notes/task7_intervention_summary.txt",
    "LSTM_seed42_before_after_notes.txt": "Task7/results/notes/LSTM_seed42_before_after_notes.txt",
}

SEARCH_DIRS = ["Task4", "Task6", "Task7", "Tests"]

def should_scan(path: Path) -> bool:
    if path.is_dir():
        return False
    if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
        return False
    return path.suffix in {".py", ".md", ".txt", ".sh", ".yaml", ".yml", ".json", ".ipynb"}

hits = []

for top in SEARCH_DIRS:
    base = ROOT / top
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for old_name, new_path in MOVED.items():
            pattern = rf"Task7/results/{re.escape(old_name)}"
            for match in re.finditer(pattern, text):
                line_no = text.count("\n", 0, match.start()) + 1
                hits.append((str(path.relative_to(ROOT)), line_no, f"Task7/results/{old_name}", new_path))

if not hits:
    print("No hardcoded old Task7 flat-path references found.")
else:
    print("Found possible outdated Task7 path references:\n")
    for file, line_no, old_ref, new_ref in hits:
        print(f"{file}:{line_no}")
        print(f"  old: {old_ref}")
        print(f"  new: {new_ref}\n")