import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_FLAT_PATHS = {
    "Task7/results/age_group_counts.csv": "Task7/results/counts/age_group_counts.csv",
    "Task7/results/race_counts.csv": "Task7/results/counts/race_counts.csv",
    "Task7/results/race_intervention_weights.csv": "Task7/results/counts/race_intervention_weights.csv",
    "Task7/results/sex_counts.csv": "Task7/results/counts/sex_counts.csv",
}

EXPECTED_WRITES = {
    Path("Task7/check_group_counts.py"): {
        "Task7/results/counts/sex_counts.csv",
        "Task7/results/counts/age_group_counts.csv",
        "Task7/results/counts/race_counts.csv",
    },
    Path("Task7/define_race_intervention.py"): {
        "Task7/results/counts/race_intervention_weights.csv",
    },
}

SEARCH_DIRS = ["Task4", "Task6", "Task7", "Tests"]
SELF_PATH = Path("Task7/check_task7_paths.py")


def should_scan(path: Path) -> bool:
    if path.is_dir():
        return False
    if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
        return False
    return path.suffix in {".py", ".md", ".txt", ".sh", ".yaml", ".yml", ".json", ".ipynb"}


def join_path(left: str, right: str) -> str:
    left = left.rstrip("/")
    right = right.lstrip("/")
    if not left:
        return right
    if not right:
        return left
    return f"{left}/{right}"


class CsvWriteCollector(ast.NodeVisitor):
    def __init__(self):
        self.values = {"REPO_ROOT": ""}
        self.writes = set()

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return node.args[0].value
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if left is None or right is None:
                return None
            return join_path(left, right)
        return None

    def visit_Assign(self, node):
        value = self._eval(node.value)
        if value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.values[target.id] = value
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to_csv":
            path_node = node.args[0] if node.args else None
            if path_node is None:
                for keyword in node.keywords:
                    if keyword.arg == "path_or_buf":
                        path_node = keyword.value
                        break
            if path_node is not None:
                path_value = self._eval(path_node)
                if path_value is not None:
                    self.writes.add(path_value.replace("\\", "/"))
        self.generic_visit(node)


def collect_writer_outputs(relative_path: Path):
    source_path = ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    collector = CsvWriteCollector()
    collector.visit(tree)
    return collector.writes


literal_hits = []

for top in SEARCH_DIRS:
    base = ROOT / top
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not should_scan(path):
            continue
        if path.relative_to(ROOT) == SELF_PATH:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for old_ref, new_ref in OLD_FLAT_PATHS.items():
            for match in re.finditer(re.escape(old_ref), text):
                line_no = text.count("\n", 0, match.start()) + 1
                literal_hits.append((path.relative_to(ROOT), line_no, old_ref, new_ref))

writer_issues = []

for relative_path, expected in EXPECTED_WRITES.items():
    actual = collect_writer_outputs(relative_path)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        writer_issues.append((relative_path, missing, unexpected))

if not literal_hits and not writer_issues:
    print("Task7 path check passed: current writers use the reorganized results layout.")
else:
    if literal_hits:
        print("Found outdated flat Task7 path references:\n")
        for file_path, line_no, old_ref, new_ref in literal_hits:
            print(f"{file_path}:{line_no}")
            print(f"  old: {old_ref}")
            print(f"  new: {new_ref}\n")

    if writer_issues:
        print("Found writer path mismatches:\n")
        for relative_path, missing, unexpected in writer_issues:
            print(f"{relative_path}")
            if missing:
                print("  missing expected writes:")
                for path in missing:
                    print(f"    {path}")
            if unexpected:
                print("  unexpected writes:")
                for path in unexpected:
                    print(f"    {path}")
            print()
