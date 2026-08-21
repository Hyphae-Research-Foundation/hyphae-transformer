"""Dependency-free, reproducible JSON and HTML run reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from celiums_rezero.lab.serialization import write_json


def render_run_report(
    output_directory: Path,
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "report.json"
    html_path = output_directory / "report.html"
    payload = {"manifest": manifest, "result": result}
    write_json(json_path, payload)

    metrics = result.get("metrics", [])
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(metric.get('name', '')))}</td>"
        f"<td>{html.escape(str(metric.get('value', '')))}</td>"
        f"<td>{html.escape(str(metric.get('unit', '')))}</td>"
        "</tr>"
        for metric in metrics
    )
    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Celiums ReZero Run {html.escape(str(result.get('run_id', '')))}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 960px; margin: 3rem auto;
        padding: 0 1rem; color: #16211d; }}
h1, h2 {{ color: #074f3b; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #b9c7c1; padding: .55rem; text-align: left; }}
pre {{ background: #eef4f1; padding: 1rem; overflow: auto; }}
</style>
<h1>Celiums ReZero Experiment</h1>
<p><strong>Run:</strong> {html.escape(str(result.get('run_id', '')))}</p>
<p><strong>Verdict:</strong> {html.escape(str(result.get('verdict', '')))}</p>
<p>{html.escape(str(result.get('summary', '')))}</p>
<h2>Metrics</h2>
<table><thead><tr><th>Name</th><th>Value</th><th>Unit</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>Manifest</h2>
<pre>{html.escape(json.dumps(manifest, sort_keys=True, indent=2))}</pre>
</html>
"""
    html_path.write_text(document)
    return json_path, html_path
