import pytest
import sys
from core.commands import run_command, CommandError, CommandResult

def test_run_command_success():
    # Nutze 'python -c print(1)' als einfachen plattformübergreifenden Befehl
    result = run_command([sys.executable, "-c", "print('hello')"])
    assert result.success
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"

def test_run_command_failure_check_true():
    with pytest.raises(CommandError) as excinfo:
        run_command([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)
    
    assert excinfo.value.result.returncode == 1
    assert "Command failed with exit code 1" in str(excinfo.value)

def test_run_command_failure_check_false():
    result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], check=False)
    assert not result.success
    assert result.returncode == 1

def test_run_command_capture_output_false():
    result = run_command([sys.executable, "-c", "print('hello')"], capture_output=False)
    assert result.success
    assert result.stdout == ""
    assert result.stderr == ""
