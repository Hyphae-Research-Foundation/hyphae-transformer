#!/usr/bin/env python3
"""Run the live Hyphae navigation canary with the calibrated ReZero pilot v2."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hyphae_minilm_gemma_navigation_canary as base

base.NAVIGATION_BUNDLE_SHA256 = "d14963e7835a81ee4ca32274d34ba5ed098270a626ba34690fb706f3465ab7ac"
base.NAVIGATION_CHECKPOINT_SHA256 = (
    "0f1f140d683df581020a39b221802f20a14ced6d4316748f70aab36ced686844"
)
base.REPORT_SCHEMA = "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v2"

if __name__ == "__main__":
    try:
        raise SystemExit(base.main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
