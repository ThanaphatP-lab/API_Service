"""Shell function tests only: never start services or modify PID files."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/model-stack.sh").read_text(encoding="utf-8")
BASH = shutil.which("bash")


@pytest.fixture(scope="module", autouse=True)
def require_bash():
    if not BASH:
        pytest.skip("Bash is not installed")
    if subprocess.run([BASH, "--version"], capture_output=True, timeout=30).returncode:
        pytest.skip("Bash is unavailable")


def shell(commands):
    defaults = "\n".join(re.findall(r'^: .*$', SCRIPT, re.M))
    arrays = "\n".join(re.findall(r'^(?:ALL_SERVICES|CORE_SERVICES)=\([\s\S]*?^\)', SCRIPT, re.M))
    functions = "\n".join(re.findall(r'^\w+\(\) \{[\s\S]*?^\}', SCRIPT, re.M))
    source = ('set -Eeuo pipefail\nROOT=/workspace/models\nCACHE_DIR=/cache\nMODEL_TMP_DIR=/tmp\n'
              + defaults + "\n" + arrays + '\nALL_WITH_DEMO=("${ALL_SERVICES[@]}" demo)\n'
              + functions + "\n" + commands)
    result = subprocess.run([BASH, "--noprofile", "--norc", "-s"],
                            input=source.encode(), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.decode().strip().splitlines()


def test_bash_syntax():
    result = subprocess.run([BASH, "-n"], input=SCRIPT.encode(), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_log_level_defaults_and_resets():
    assert shell('parse_start_options --log-level debug; echo "$STACK_LOG_LEVEL"; parse_start_options; echo "$STACK_LOG_LEVEL"') == ["debug", "info"]


@pytest.mark.parametrize("options", ["--log-level", "--log-level invalid", "--unknown debug"])
def test_invalid_log_options_rejected(options):
    assert shell(f'if parse_start_options {options}; then echo accepted; else echo rejected; fi')[-1] == "rejected"


def test_restart_and_uvicorn_pass_selected_level():
    assert 'start "$target" --log-level "$STACK_LOG_LEVEL"' in SCRIPT
    assert '--workers 1 --log-level "$STACK_LOG_LEVEL"' in SCRIPT


@pytest.mark.parametrize("profile", ["all", "core-stack", "ocr-custom-stack", "layout-stack", "table-stack"])
def test_profiles_start_exactly_one_detector(profile):
    names = shell(f"start_target_services {profile}")[0].split()
    assert names.count("detection") == 1
    assert not {"det-v5", "det-v6"}.intersection(names)


def test_detection_uses_registry_without_fixed_name_or_directory():
    lines = shell('''
DET_MODEL_NAME=old-model
DET_MODEL_DIR=/old-weights
unset MODEL_VARIANTS_CONFIG DET_MODEL_VERSION
export_service_environment detection
printf '%s\\n' "${DET_MODEL_NAME-unset}" "${DET_MODEL_DIR-unset}" "$DET_MODEL_VERSION" "$MODEL_VARIANTS_CONFIG"
service_module detection
service_port detection
''')
    assert lines == ["unset", "unset", "v5", "/workspace/models/model_variants.json",
                     "services.text_det.main:app", "8002"]


def test_custom_port_and_registry_are_preserved():
    lines = shell('''
DETECTION_PORT=8123
MODEL_VARIANTS_CONFIG=/custom/variants.json
DET_MODEL_VERSION=v6
unset TEXT_DETECTION_URL
export_service_environment detection
printf '%s\\n' "$TEXT_DETECTION_URL" "$DET_SERVICE_URL" "$MODEL_VARIANTS_CONFIG" "$DET_MODEL_VERSION"
''')
    assert lines == ["http://127.0.0.1:8123"] * 2 + ["/custom/variants.json", "v6"]


@pytest.mark.parametrize("name", ["det-v5", "det-v6"])
def test_legacy_names_are_not_valid_targets(name):
    assert shell(f'if target_services {name}; then exit 9; fi\necho rejected\n') == ["rejected"]
