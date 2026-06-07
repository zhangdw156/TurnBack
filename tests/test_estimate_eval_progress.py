from scripts.estimate_eval_progress import ModelProgress, format_progress_bar, render_table


def test_format_progress_bar_clamps_and_formats_percentage():
    assert format_progress_bar(-1.0).plain == "[░░░░░░░░░░░░░░░░░░] 0.00%"
    assert format_progress_bar(0.5, width=4).plain == "[██░░] 50.00%"
    assert format_progress_bar(2.0, width=4).plain == "[████] 100.00%"


def test_render_table_accepts_progress_text_without_rich_progressbar_type_error(capsys):
    item = ModelProgress(
        model="qwen3-4b-thinking-2507",
        result_path="results/qwen3-4b-thinking-2507/turnback_10pct.jsonl",
        completed=24,
        observed=3900,
        bad_jsonl_lines=0,
        pending=3876,
        total=3900,
        progress=24 / 3900,
        mean_similarity_done=44.8069,
        by_city={},
        by_difficulty={},
    )

    render_table([item])

    output = capsys.readouterr().out
    assert "TurnBack evaluation progress" in output
