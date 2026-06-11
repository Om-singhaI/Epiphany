"""Findings report writer — Epiphany's human-readable local output.

Every cycle produces a real, self-contained report on disk (Markdown + HTML) so
anyone can read what the agent discovered without a dashboard, a GitLab account,
or any cloud service. The report ties together the four real artifacts of a
cycle: the dataset profile, the hypothesis, the statistical test, and (when
significant) the trained model's measured performance.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("epiphany.report")


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", str(value)).strip("_").lower() or "finding"


def build_markdown(
    *,
    goal: str,
    profile: dict[str, Any],
    hypothesis: dict[str, Any],
    validation: dict[str, Any],
    model: dict[str, Any] | None,
    merge_request: dict[str, Any] | None,
    created_at: str,
) -> str:
    """Render the findings report as Markdown from real cycle artifacts."""
    feature = hypothesis.get("feature", "?")
    target = hypothesis.get("target", "?")
    sig = validation.get("is_significant")
    verdict = "✅ Significant" if sig else "❌ Not significant"

    lines: list[str] = []
    lines.append(f"# Epiphany findings — {target}")
    lines.append("")
    lines.append(f"*Generated {created_at} — data mode: **{profile.get('mode')}** "
                 f"({profile.get('row_count'):,} rows from `{profile.get('source')}`)*")
    lines.append("")
    lines.append(f"**Mission:** {goal}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append(f"> {hypothesis.get('statement', '(none)')}")
    lines.append("")
    lines.append(f"- **Driver (feature):** `{feature}`")
    lines.append(f"- **Outcome (target):** `{target}`")
    if hypothesis.get("rationale"):
        lines.append(f"- **Rationale:** {hypothesis['rationale']}")
    lines.append("")

    lines.append("## Statistical validation")
    lines.append("")
    lines.append(f"**Verdict: {verdict}** (α = {validation.get('alpha', 0.05)})")
    lines.append("")
    lines.append(f"- **Test:** {validation.get('test')}")
    stat_name = validation.get("statistic_name") or "statistic"
    lines.append(f"- **{stat_name}:** {validation.get('statistic')}")
    lines.append(f"- **p-value:** {validation.get('p_value')}")
    if validation.get("effect_size") is not None:
        lines.append(
            f"- **Effect size ({validation.get('effect_name')}):** "
            f"{validation.get('effect_size')}"
        )
    lines.append(f"- **Sample size:** {validation.get('sample_size'):,}")
    if validation.get("summary"):
        lines.append("")
        lines.append(f"_{validation['summary']}_")
    lines.append("")

    if model:
        lines.append("## Trained model")
        lines.append("")
        lines.append(f"- **Task:** {model.get('task')}")
        lines.append(f"- **Algorithm:** {model.get('algorithm')}")
        lines.append(f"- **Features used:** {model.get('n_features')}")
        metrics = model.get("metrics", {})
        if metrics:
            lines.append("- **Measured performance:**")
            for k, v in metrics.items():
                lines.append(f"    - {k}: {v}")
        importances = model.get("feature_importances", [])
        if importances:
            lines.append("")
            lines.append("### Top drivers (model feature importance)")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---|---|")
            for row in importances[:10]:
                lines.append(f"| `{row['feature']}` | {row['importance']} |")
        if model.get("model_path"):
            lines.append("")
            lines.append(f"_Model artifact: `{model['model_path']}`_")
        lines.append("")

    if merge_request:
        lines.append("## Deployment")
        lines.append("")
        url = merge_request.get("web_url") or merge_request.get("url")
        lines.append(f"- **Merge request:** [{merge_request.get('title')}]({url})")
        lines.append(f"- **Branch:** `{merge_request.get('source_branch')}`")
        lines.append(f"- **Mode:** {merge_request.get('mode')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Authored autonomously by Epiphany — an AI data scientist.*")
    return "\n".join(lines)


def _markdown_to_html(md: str, title: str) -> str:
    """Minimal, dependency-free Markdown→HTML for the saved report."""
    html_lines: list[str] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("| "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            tag = "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if line.startswith("# "):
            html_lines.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.startswith("    - "):
            html_lines.append(f"<li style='margin-left:1.5em'>{_inline(line[6:])}</li>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{_inline(line[2:])}</li>")
        elif line == "---":
            html_lines.append("<hr>")
        elif line:
            html_lines.append(f"<p>{_inline(line)}</p>")
    if in_table:
        html_lines.append("</table>")
    body = "\n".join(html_lines)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1a1a2e}"
        "h1,h2,h3{color:#0f3460}blockquote{border-left:4px solid #16c79a;"
        "margin:0;padding:.5em 1em;background:#f4f7f6}"
        "table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:6px}"
        "code{background:#eef;padding:1px 4px;border-radius:3px}</style></head>"
        f"<body>{body}</body></html>"
    )


def _inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", text)
    return text


def write_report(
    reports_dir: str,
    *,
    goal: str,
    profile: dict[str, Any],
    hypothesis: dict[str, Any],
    validation: dict[str, Any],
    model: dict[str, Any] | None = None,
    merge_request: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write the Markdown + HTML report to ``reports_dir``; return file paths."""
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = build_markdown(
        goal=goal,
        profile=profile,
        hypothesis=hypothesis,
        validation=validation,
        model=model,
        merge_request=merge_request,
        created_at=created_at,
    )
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{_slug(hypothesis.get('target', 'finding'))}"
    md_path = out / f"{base}.md"
    html_path = out / f"{base}.html"
    paths: dict[str, str] = {}
    try:
        md_path.write_text(md)
        paths["markdown"] = str(md_path)
        title = f"Epiphany — {hypothesis.get('target', 'finding')}"
        html_path.write_text(_markdown_to_html(md, title))
        paths["html"] = str(html_path)
    except Exception as exc:  # noqa: BLE001 - never fail a cycle on report IO
        logger.warning("Could not write report: %s", exc)
    return paths
