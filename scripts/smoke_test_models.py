"""Check real EEGNet/EEGNeX forward, backward and optimizer updates on train data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline_example import make_parser, run_cases, write_report


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/integration/smoke_test_models.json")
    args = parser.parse_args()
    records = run_cases(args)
    write_report(args.output, records, "One train batch per dataset/subject/model; integration check only")


if __name__ == "__main__":
    main()
