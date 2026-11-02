"""prepare.fail_fast: a meds_reader failure must end the run with an error, not hang it."""

import pathlib
import subprocess
import sys
import textwrap

import pytest

TRAINING_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "femr" / "training"

SCRIPT = textwrap.dedent(
    """
    import operator, sys
    sys.path.insert(0, {training_dir!r})
    import meds_reader
    from prepare import fail_fast

    if __name__ == "__main__":  # meds_reader spawns its workers, which import this file
        with meds_reader.SubjectDatabase({database!r}, num_threads=2) as database, fail_fast(database, poll_seconds=1):
            if {mode!r} == "worker":
                list(database.map(int))  # int(<generator>) raises inside the worker processes
            else:
                results = database.map(operator.length_hint)
                next(iter(results))
                raise ValueError("parent failed while results were pending")
        print("not reached")
    """
)


@pytest.mark.parametrize("mode", ["worker", "parent"])
def test_failure_exits_with_error_instead_of_hanging(toy_database, tmp_path, mode):
    script = tmp_path / "run.py"
    script.write_text(SCRIPT.format(training_dir=str(TRAINING_DIR), database=toy_database, mode=mode))

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)

    assert result.returncode == 1, result.stderr[-2000:]
    assert "not reached" not in result.stdout
    expected = "died (exit code" if mode == "worker" else "prepare.py failed"
    assert expected in result.stderr
