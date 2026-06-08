from __future__ import annotations

import re

from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+.+$")
FORBIDDEN_FRAMEWORK_MEMORY_SECTION_LABELS = frozenset(
    {
        "Formulas Used Across Document",
        "Important Assets",
        "Section Map",
    }
)


def apply_markdown_edits(existing_md: str, edits: list[MarkdownMemoryEdit]) -> str:
    updated_md = existing_md
    for edit in edits:
        if edit.edit_type == "no_change":
            continue

        if edit.section_heading is None:
            raise ValueError(f"section_heading is required for {edit.edit_type}")

        if edit.edit_type == "append_new_section":
            updated_md = _append_new_section(updated_md, edit.section_heading, edit.content_md)
        elif edit.edit_type == "append_to_section":
            updated_md = _append_to_section(updated_md, edit.section_heading, edit.content_md)
        elif edit.edit_type == "replace_section":
            updated_md = _replace_section(updated_md, edit.section_heading, edit.content_md)

    return updated_md


def apply_framework_memory_edits(existing_md: str, edits: list[MarkdownMemoryEdit]) -> str:
    cleaned_md = _remove_sections_by_heading_labels(existing_md, FORBIDDEN_FRAMEWORK_MEMORY_SECTION_LABELS)
    allowed_edits = [
        edit for edit in edits if not _is_forbidden_framework_memory_edit(edit)
    ]
    updated_md = apply_markdown_edits(cleaned_md, allowed_edits)
    return _remove_sections_by_heading_labels(updated_md, FORBIDDEN_FRAMEWORK_MEMORY_SECTION_LABELS)


def apply_short_term_memory_edits(existing_md: str, edits: list[MarkdownMemoryEdit]) -> str:
    return apply_markdown_edits(existing_md, edits)


def _append_new_section(existing_md: str, section_heading: str, content_md: str) -> str:
    base = existing_md.rstrip()
    heading = _normalize_section_heading(section_heading)
    content = content_md.strip()
    return f"{base}\n\n{heading}\n{content}\n"


def _append_to_section(existing_md: str, section_heading: str, content_md: str) -> str:
    lines = existing_md.splitlines()
    heading_index = _find_heading_index(lines, section_heading)
    if heading_index is None:
        return _append_new_section(existing_md, section_heading, content_md)

    lines[heading_index] = _normalize_section_heading(lines[heading_index])
    end_index = _find_section_end_index(lines, heading_index)
    before = _rstrip_blank_lines(lines[:end_index])
    after = lines[end_index:]
    content_lines = ["", *content_md.strip().splitlines(), ""]
    return _join_lines([*before, *content_lines, *after])


def _replace_section(existing_md: str, section_heading: str, content_md: str) -> str:
    lines = existing_md.splitlines()
    heading_index = _find_heading_index(lines, section_heading)
    if heading_index is None:
        return _append_new_section(existing_md, section_heading, content_md)

    lines[heading_index] = _normalize_section_heading(lines[heading_index])
    end_index = _find_section_end_index(lines, heading_index)
    before = lines[: heading_index + 1]
    after = lines[end_index:]
    replacement = [*content_md.strip().splitlines(), ""]
    return _join_lines([*before, *replacement, *after])


def _find_heading_index(lines: list[str], section_heading: str) -> int | None:
    target_label = _heading_label(section_heading)
    for index, line in enumerate(lines):
        if _heading_label(line) == target_label:
            return index
    return None


def _normalize_section_heading(section_heading: str) -> str:
    stripped = section_heading.strip()
    if _is_heading(stripped):
        return stripped
    return f"## {stripped.lstrip('#').strip()}"


def _heading_label(line: str) -> str:
    stripped = line.strip()
    if _is_heading(stripped):
        return re.sub(r"^#{1,6}\s+", "", stripped).strip()
    return stripped.lstrip("#").strip()


def _find_section_end_index(lines: list[str], heading_index: int) -> int:
    current_level = _heading_level(lines[heading_index])
    for index in range(heading_index + 1, len(lines)):
        if _is_heading(lines[index]) and _heading_level(lines[index]) <= current_level:
            return index
    return len(lines)


def _is_forbidden_framework_memory_edit(edit: MarkdownMemoryEdit) -> bool:
    if edit.section_heading is None:
        return False
    return _heading_label(edit.section_heading) in FORBIDDEN_FRAMEWORK_MEMORY_SECTION_LABELS


def _remove_sections_by_heading_labels(existing_md: str, forbidden_labels: frozenset[str]) -> str:
    lines = existing_md.splitlines()
    filtered_lines: list[str] = []
    index = 0
    while index < len(lines):
        if _heading_label(lines[index]) in forbidden_labels and (
            _is_heading(lines[index]) or lines[index].strip() in forbidden_labels
        ):
            index = _find_section_end_index_for_cleanup(lines, index)
            continue
        filtered_lines.append(lines[index])
        index += 1
    return _join_lines(_rstrip_blank_lines(filtered_lines))


def _find_section_end_index_for_cleanup(lines: list[str], heading_index: int) -> int:
    current_level = _heading_level(lines[heading_index]) if _is_heading(lines[heading_index]) else 2
    for index in range(heading_index + 1, len(lines)):
        if _is_heading(lines[index]) and _heading_level(lines[index]) <= current_level:
            return index
    return len(lines)


def _is_heading(line: str) -> bool:
    return HEADING_PATTERN.match(line.strip()) is not None


def _heading_level(line: str) -> int:
    match = HEADING_PATTERN.match(line.strip())
    if match is None:
        raise ValueError(f"Not a Markdown heading: {line}")
    return len(match.group(1))


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _rstrip_blank_lines(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and trimmed[-1].strip() == "":
        trimmed.pop()
    return trimmed
