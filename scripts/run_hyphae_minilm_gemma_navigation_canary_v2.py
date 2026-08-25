#!/usr/bin/env python3
"""Run the live Hyphae navigation canary with the calibrated ReZero pilot v2."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hyphae_minilm_gemma_navigation_canary as base

base.NAVIGATION_BUNDLE_SHA256 = "NAV2_BUNDLE_SHA256"
base.NAVIGATION_CHECKPOINT_SHA256 = "NAV2_CHECKPOINT_SHA256"
base.REPORT_SCHEMA = "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v2"

if __name__ == "__main__":
    raise SystemExit(base.main())
