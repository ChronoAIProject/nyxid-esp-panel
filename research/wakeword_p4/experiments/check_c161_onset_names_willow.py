"""Replay the official Willow comparator on the frozen onset-name challenge."""
from pathlib import Path
import sys
from unittest.mock import patch

import check_c161_melo_address_grammar as melo
import check_c161_onset_names_melo as challenge


def main():
    action = sys.argv.pop(1)
    runs = challenge.WORK / "willow"
    if action == "case":
        sys.path.insert(0, str((melo.ROOT / "qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        runner_path = Path(official.__file__).relative_to(Path.cwd())
        with patch.multiple(official, PLAN=challenge.PLAN, __file__=str(runner_path)):
            official.case()
    elif action == "willow":
        import run_c161_willow_matched_scenes_v1 as runner
        with patch.multiple(runner, PLAN=challenge.PLAN, OUT=runs, __file__=str(Path(__file__).resolve())):
            runner.willow()
    elif action == "summarize":
        with patch.multiple(melo, PLAN=challenge.PLAN, STEM=challenge.STEM, RUNS=runs):
            melo.summarize_willow()
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
