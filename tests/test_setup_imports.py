from src.eval.result_writer import RESULT_FIELDNAMES, ResultRecord
from src.utils.paths import PROJECT_ROOT


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_result_record_schema_contains_required_fields():
    record = ResultRecord(dataset="setup", method="test", seed=0)
    row = record.to_csv_row()

    for field in RESULT_FIELDNAMES:
        assert field in row

