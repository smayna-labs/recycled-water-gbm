"""
pipeline.py: reproduce every number and figure in the paper.

    python src/pipeline.py
"""
import time
import data_prep, evaluate, diagnostics


def banner(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


if __name__ == "__main__":
    t0 = time.time()
    banner("1. Data preparation"); data_prep.main()
    banner("2. Rolling-origin evaluation, tests, figures"); evaluate.main()
    banner("3. Diagnostics (facility selection, DSRSD rainfall)"); diagnostics.main()
    banner(f"Done in {time.time() - t0:.0f}s")
