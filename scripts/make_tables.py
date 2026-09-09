#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"
CORPUS = {"toolace": "ToolACE", "retool": "ReTool", "glaive": "Glaive-FC"}
POLICY = {
    "bfd": "BFD",
    "bfd_split": "BFD-split",
    "wrapped_single_batch": "Wrapped (intended)",
    "keep_end": "Keep-end (historical)",
    "source_target_round": "Complete rounds",
    "single_target_closure": "One example per target",
    "grouped_closure": "Grouped",
}


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}\\%"


def save(frame: pd.DataFrame, name: str, column_format: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    latex = frame.to_latex(index=False, escape=False, column_format=column_format)
    (OUT / f"{name}.tex").write_text(latex)


def contemporary() -> None:
    frames = [
        pd.read_csv(ROOT / "analysis/corpus_audit/toolace_retool/incidence.csv"),
        pd.read_csv(ROOT / "analysis/corpus_audit/glaive/incidence.csv"),
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data[
        (data.stratum == "all_admitted")
        & (data.max_length == 512)
        & data.policy.isin(["bfd", "bfd_split", "wrapped_single_batch", "keep_end"])
    ].copy()
    rows = []
    for row in data.sort_values(["corpus", "policy"]).itertuples():
        rows.append(
            {
                "Corpus": CORPUS[row.corpus],
                "Policy": POLICY[row.policy],
                "Matched targets retained": f"{row.trained_units:,}/{row.oracle_units:,}",
                "Separated": f"{row.unsupported_units:,}/{row.trained_units:,} ({percent(row.conditional_prevalence)})",
                "Affected conversations": f"{row.affected_conversations:,}/{row.total_conversations:,} ({percent(row.conversation_incidence)})",
                "Affected assistant tokens": percent(row.nominal_objective_exposure, 3),
                "Assistant supervision retained": percent(row.supervised_token_retention),
            }
        )
    save(pd.DataFrame(rows), "table1_contemporary_512")


def case_sensitive() -> None:
    data = pd.read_csv(ROOT / "analysis/case_sensitive/case_sensitive_512.csv")
    rows = []
    for row in data.sort_values(["corpus", "policy"]).itertuples():
        rows.append(
            {
                "Corpus": CORPUS[row.corpus],
                "Policy": POLICY[row.policy],
                "Separated / retained": f"{row.unsupported_units_strict:,}/{row.trained_units_strict:,}",
                "Separation rate": percent(row.conditional_prevalence_strict),
                "Affected conversations": f"{row.affected_conversations_strict:,}/{row.total_conversations_strict:,}",
                "Affected assistant tokens": percent(row.nominal_objective_exposure_strict, 3),
                "Primary rate": percent(row.conditional_prevalence_primary),
                "$\\Delta$": f"{row.delta_conditional_percentage_points:+.2f} pp",
            }
        )
    save(pd.DataFrame(rows), "tableS2_case_sensitive_512")

    complete = pd.read_csv(ROOT / "analysis/case_sensitive/case_sensitive_all_budgets.csv")
    complete = complete[complete.stratum == "all_admitted"]
    rows = []
    for row in complete.sort_values(["corpus", "policy", "max_length"]).itertuples():
        rows.append(
            {
                "Corpus": CORPUS[row.corpus],
                "Policy": POLICY[row.policy],
                "Budget": f"{row.max_length:,}",
                "Separated / retained": f"{row.unsupported_units_strict:,}/{row.trained_units_strict:,}",
                "Separation rate": percent(row.conditional_prevalence_strict),
                "Affected conversations": f"{row.affected_conversations_strict:,}/{row.total_conversations_strict:,}",
                "Affected assistant tokens": percent(row.nominal_objective_exposure_strict, 3),
                "Primary rate": percent(row.conditional_prevalence_primary),
                "$\\Delta$": f"{row.delta_conditional_percentage_points:+.2f} pp",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tableS3_case_sensitive_all_budgets.csv", index=False)
    latex = frame.to_latex(
        index=False,
        escape=False,
        longtable=True,
        column_format="lllrrrrrr",
        caption="Case-sensitive matching at every evaluated token budget.",
        label="tab:case-all",
    )
    for label, wrapped in {
        "Separated / retained": r"\shortstack{Separated /\\retained}",
        "Separation rate": r"\shortstack{Separation\\rate}",
        "Affected conversations": r"\shortstack{Affected\\conversations}",
        "Affected assistant tokens": r"\shortstack{Affected\\assistant tokens}",
        "Primary rate": r"\shortstack{Primary\\rate}",
    }.items():
        latex = latex.replace(label, wrapped)
    (OUT / "tableS3_case_sensitive_all_budgets.tex").write_text(latex)


def materialization() -> None:
    data = pd.read_csv(ROOT / "outputs/materialization/summary.csv")
    data = data[
        (data.max_length == 512)
        & data.policy.isin(["source_target_round", "single_target_closure", "grouped_closure"])
    ]
    rows = []
    for row in data.sort_values(["corpus", "policy"]).itertuples():
        rows.append(
            {
                "Corpus": CORPUS[row.corpus],
                "Reconstruction": POLICY[row.policy],
                "Matched targets retained": f"{row.retained_targets:,}/{row.valid_targets:,}",
                "Retention": percent(row.target_retention),
                "Target supervision retained": percent(row.target_supervision_ratio),
                "Input-token ratio": percent(row.input_token_ratio),
                "Examples / conversation": f"{row.examples_per_conversation:.3f}",
                "Duplicate source tokens": f"{row.duplicate_source_tokens:,}",
            }
        )
    save(pd.DataFrame(rows), "table2_materialization_512")


def schema() -> None:
    data = pd.read_csv(ROOT / "outputs/schema_repair/summary.csv")
    pivot = data.pivot(index=["corpus", "mode"], columns="max_length", values="repair_rate").reset_index()
    pivot["Corpus"] = pivot.corpus.map(CORPUS)
    pivot["Schema content"] = pivot["mode"].map(
        {"whole_schema": "Full schema", "invoked_tool": "Invoked tool", "parameter_slice": "Referenced parameters"}
    )
    pivot["order"] = pivot["mode"].map({"whole_schema": 0, "invoked_tool": 1, "parameter_slice": 2})
    pivot = pivot.sort_values(["corpus", "order"])
    output = pivot[["Corpus", "Schema content", 256, 512, 1024, 2048]].copy()
    output.columns = ["Corpus", "Schema content", "256", "512", "1,024", "2,048"]
    for column in ["256", "512", "1,024", "2,048"]:
        output[column] = output[column].map(percent)
    save(output, "table3_schema_repair")


def provenance() -> None:
    manifest = json.loads((ROOT / "data/processed/glaive_manifest.json").read_text())
    output = pd.DataFrame(
        [
            {
                "Dataset": "glaiveai/glaive-function-calling-v2",
                "Revision": "e7f4b6456019",
                "License": "Apache-2.0",
                "Release rows": f"{manifest['source_rows']:,}",
                "Eligible rows": f"{manifest['eligible_rows']:,}",
                "Evaluation sample": f"{manifest['selected_rows']:,}",
                "Candidate relations": f"{manifest['selected_edges']:,}",
            }
        ]
    )
    save(output, "tableS1_glaive_provenance")


def model_study() -> None:
    path = ROOT / "analysis/model_study/aggregate.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).set_index(["split", "corruption"])
    rows = []
    base_test = ROOT / "outputs/model_study/eval_test_base/summary.csv"
    base_conflict = ROOT / "outputs/model_study/eval_conflict_base/summary.csv"
    if base_test.exists() and base_conflict.exists():
        test = pd.read_csv(base_test).set_index("family").loc["ALL"]
        conflict = pd.read_csv(base_conflict).set_index("family").loc["ALL"]
        rows.append(
            {
                "Condition": "Unadapted base",
                "Clean exact": f"{100*test.exact:.2f}\\%",
                "Clean value NLL": f"{test.value_token_nll:.3f}",
                "Conflict exact": f"{100*conflict.exact:.2f}\\%",
                "Conflict stale copy": f"{100*conflict.copied_stale:.2f}\\%",
            }
        )
    for corruption in [0, 25, 50, 100]:
        test = data.loc[("test", corruption)]
        conflict = data.loc[("conflict", corruption)]
        rows.append(
            {
                "Condition": f"{corruption}\\% removed",
                "Clean exact": f"{100*test.exact_mean:.2f} $\\pm$ {100*test.exact_seed_sd:.2f}\\%",
                "Clean value NLL": f"{test.value_token_nll_mean:.3f} $\\pm$ {test.value_token_nll_seed_sd:.3f}",
                "Conflict exact": f"{100*conflict.exact_mean:.2f} $\\pm$ {100*conflict.exact_seed_sd:.2f}\\%",
                "Conflict stale copy": f"{100*conflict.copied_stale_mean:.2f} $\\pm$ {100*conflict.copied_stale_seed_sd:.2f}\\%",
            }
        )
    save(pd.DataFrame(rows), "table4_model_study")


def model_families() -> None:
    path = ROOT / "analysis/model_study/family_aggregate.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).set_index(["split", "family", "corruption"])
    rows = []
    for family in ["order", "error", "inventory", "route", "ticket", "checksum"]:
        test_0 = data.loc[("test", family, 0)]
        test_100 = data.loc[("test", family, 100)]
        conflict_0 = data.loc[("conflict", family, 0)]
        conflict_100 = data.loc[("conflict", family, 100)]
        rows.append(
            {
                "Family": family.title(),
                "Clean exact, 0\\% removed": percent(test_0.exact_mean),
                "Clean exact, 100\\% removed": percent(test_100.exact_mean),
                "Clean NLL, 0\\% removed": f"{test_0.value_token_nll_mean:.3f}",
                "Clean NLL, 100\\% removed": f"{test_100.value_token_nll_mean:.3f}",
                "Conflict exact, 0\\% removed": percent(conflict_0.exact_mean),
                "Conflict exact, 100\\% removed": percent(conflict_100.exact_mean),
                "Conflict stale copy, 100\\% removed": percent(conflict_100.copied_stale_mean),
            }
        )
    save(pd.DataFrame(rows), "tableS2_model_families")


def main() -> None:
    contemporary()
    case_sensitive()
    materialization()
    schema()
    provenance()
    model_study()
    model_families()
    manifest = {path.name: path.stat().st_size for path in sorted(OUT.glob("table*"))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
