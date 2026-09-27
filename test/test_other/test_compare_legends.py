from unittest.mock import patch

import numpy as np

from script.imgs.compare import create_legend as legends


def test_column_order_preserves_displayed_rows():
    for count, ncol in ((2, 2), (6, 3), (7, 4), (8, 4)):
        entries = list(range(count))
        ordered = legends.column_major_entries(entries, ncol)
        columns = np.array_split(ordered, ncol)
        displayed = [
            column[row]
            for row in range(max(map(len, columns)))
            for column in columns
            if row < len(column)
        ]
        assert displayed == entries


def test_standard_legends_use_at_most_four_columns():
    with patch.object(legends, "draw_legend") as draw:
        for filename, entries in legends.LEGENDS.items():
            legends.create_legend(filename, entries)
            assert draw.call_args.kwargs["ncol"] == min(len(entries), 4)
            assert [entry[0] for entry in draw.call_args.args[1]] == [
                label for _, label in entries
            ]


def test_sensitivity_groups_methods_by_row():
    with patch.object(legends, "draw_legend") as draw:
        legends.create_sensitivity_legend()
    assert draw.call_args.kwargs["ncol"] == 3
    assert [entry[0] for entry in draw.call_args.args[1]] == [
        f"{method} ({setting})"
        for method in ("Ours", "Ours-AR")
        for setting in ("underestimated", "nominal", "overestimated")
    ]
