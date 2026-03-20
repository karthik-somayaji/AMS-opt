"""Convenience wrapper to run QA filtering from the repo root.

This forwards execution to `QA_Task/filtered_QA/filter_with_llmbo.py`.
"""

import os
import runpy


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(repo_root, "QA_Task", "filtered_QA", "filter_with_llmbo.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
