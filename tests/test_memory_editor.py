from vision_indexer.memory.memory_editor import apply_framework_memory_edits
from vision_indexer.memory.memory_editor import apply_markdown_edits
from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit


def test_apply_markdown_edits_no_change_keeps_existing_memory() -> None:
    existing = "# Framework Memory\n\n## Document Identity\nExisting.\n"

    result = apply_markdown_edits(
        existing,
        [MarkdownMemoryEdit(edit_type="no_change", section_heading=None, content_md="")],
    )

    assert result == existing


def test_apply_markdown_edits_appends_new_section() -> None:
    existing = "# Framework Memory\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_new_section",
                section_heading="## Document Identity",
                content_md="ResNet paper.",
            )
        ],
    )

    assert result.endswith("## Document Identity\nResNet paper.\n")


def test_apply_markdown_edits_appends_to_existing_section_before_next_peer_heading() -> None:
    existing = "# Memory\n\n## Current Section\nPage one.\n\n## Next Section\nKeep this.\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Current Section",
                content_md="Page two.",
            )
        ],
    )

    assert "## Current Section\nPage one.\n\nPage two.\n\n## Next Section\nKeep this." in result


def test_apply_markdown_edits_replaces_existing_section_body() -> None:
    existing = "# Memory\n\n## Current Section\nOld body.\n\n### Child\nOld child.\n\n## Next Section\nKeep this.\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="replace_section",
                section_heading="## Current Section",
                content_md="New body.",
            )
        ],
    )

    assert "## Current Section\nNew body.\n\n## Next Section\nKeep this." in result
    assert "Old body" not in result
    assert "Old child" not in result


def test_apply_markdown_edits_appends_missing_section_for_section_edits() -> None:
    existing = "# Memory\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Missing",
                content_md="Created at end.",
            )
        ],
    )

    assert result.endswith("## Missing\nCreated at end.\n")


def test_apply_markdown_edits_normalizes_bare_new_section_heading() -> None:
    existing = "# Framework Memory\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_new_section",
                section_heading="Document Identity",
                content_md="ResNet paper.",
            )
        ],
    )

    assert "## Document Identity\nResNet paper.\n" in result


def test_apply_markdown_edits_repairs_existing_bare_heading_before_appending() -> None:
    existing = "# Framework Memory\n\nCore Concepts and Early Section Map\n- Page one.\n"

    result = apply_markdown_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="Core Concepts and Early Section Map",
                content_md="- Page two.",
            )
        ],
    )

    assert "## Core Concepts and Early Section Map\n- Page one.\n\n- Page two.\n" in result
    assert "\nCore Concepts and Early Section Map\n" not in result


def test_apply_framework_memory_edits_strips_forbidden_existing_sections() -> None:
    existing = """# Framework Memory

## Document Identity
- Existing identity.

## Formulas Used Across Document
- H(x) = F(x) + x.

## Important Assets
- Table 1 describes architectures.

## Section Map
- Page 1 has the abstract.
"""

    result = apply_framework_memory_edits(existing, [])

    assert "## Document Identity" in result
    assert "## Section Map" not in result
    assert "## Formulas Used Across Document" not in result
    assert "## Important Assets" not in result
    assert "H(x) = F(x) + x" not in result
    assert "Table 1 describes architectures" not in result
    assert "Page 1 has the abstract" not in result


def test_apply_framework_memory_edits_ignores_forbidden_formula_section_edit() -> None:
    existing = "# Framework Memory\n\n## Section Map\n- Page 1 has the abstract.\n"

    result = apply_framework_memory_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_new_section",
                section_heading="## Formulas Used Across Document",
                content_md="- H(x) = F(x) + x.",
            )
        ],
    )

    assert "## Section Map" not in result
    assert "## Formulas Used Across Document" not in result
    assert "H(x) = F(x) + x" not in result


def test_apply_framework_memory_edits_ignores_forbidden_assets_section_edit() -> None:
    existing = "# Framework Memory\n\n## Section Map\n- Page 1 has the abstract.\n"

    result = apply_framework_memory_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Important Assets",
                content_md="- Figure 3 compares architectures.",
            )
        ],
    )

    assert "## Section Map" not in result
    assert "## Important Assets" not in result
    assert "Figure 3 compares architectures" not in result


def test_apply_framework_memory_edits_ignores_forbidden_section_map_edit() -> None:
    existing = "# Framework Memory\n\n## Section Map\n- Page 1 has the abstract.\n"

    result = apply_framework_memory_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Section Map",
                content_md="- Page 2 begins related work.",
            )
        ],
    )

    assert "## Section Map" not in result
    assert "- Page 1 has the abstract." not in result
    assert "- Page 2 begins related work." not in result


def test_apply_framework_memory_edits_still_applies_allowed_recurring_concepts_edit() -> None:
    existing = "# Framework Memory\n\n## Recurring Concepts\n- Residual learning.\n"

    result = apply_framework_memory_edits(
        existing,
        [
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Recurring Concepts",
                content_md="- Identity shortcut.",
            )
        ],
    )

    assert "## Recurring Concepts" in result
    assert "- Residual learning." in result
    assert "- Identity shortcut." in result
