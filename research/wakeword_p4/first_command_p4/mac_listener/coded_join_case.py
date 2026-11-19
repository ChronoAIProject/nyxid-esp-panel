"""Run the pinned official DEFAULT Willow replay on one close-join command."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
sys.path.insert(0, str((ROOT/"qemu_hey_reference_v2").resolve()))
import run_willow_mac_matched_prime_v1 as official

if __name__ == "__main__":
    assert sys.argv.pop(1) == "case"
    official.PLAN = ROOT/"results/4090-c161-coded-close-join-plan.json"
    official.__file__ = str(Path(__file__).resolve())
    official.case()
