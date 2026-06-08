from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from vision_indexer.schemas.page_output import PageIndexOutput
from vision_indexer.schemas.topic_index import TopicIndexOutput


SYSTEM_PROMPT = """You are building topic_index.json, the final navigation map for a PDF.

Use the provided current batch page outputs JSON only. Do not ask for page images. Do not invent pages that are not present.

Your job:
- update the previous topic index with the current batch page outputs
- merge duplicate or similar page-level topics into clean final topics
- create stable topic IDs such as T001, T002, T003
- choose only the best pages under primary_pages
- put assets inside the primary_pages item where the asset appears
- include non-index-worthy pages only in unindexed_pages unless they contain useful routing content
- use latest framework memory only as document context, not as final navigation output
- keep old topics unless the current batch clearly improves, merges, or removes them
- do not restart from scratch after batch 1

Return exactly one TopicIndexOutput object:
- document
- topics
- unindexed_pages

Each topic must contain:
- topic_id
- topic_name
- topic_description
- primary_pages

topic_description rules:
- factual only, with no vague wording or filler
- explain what the topic is and why it matters inside this document
- mention important methods, datasets, equations, results, figures, or tables when they help route a user query
- include enough concrete context for an AI agent to choose the right page without reading every page
- do not use marketing-style language, generic praise, or broad claims unsupported by the page JSON

Each primary_pages item must contain:
- page_number
- page_image_path
- description
- assets

Each nested asset must contain:
- asset_id
- asset_type
- asset_name
- asset_description

Do not use key_assets.
Do not use why_this_page.
Do not use supporting_pages.
Do not use aliases.
Do not use page-level raw sections or titles.
Do not use section_path or source_section_id.
Do not resend or require old page JSON files.

The result must be compact enough for routing but descriptive enough for an agent to choose the right page."""


def build_topic_index_messages(
    page_outputs: Sequence[PageIndexOutput],
    source_pdf_path: Path,
    previous_topic_index: TopicIndexOutput | None = None,
    framework_memory_md: str = "",
    batch_number: int = 1,
    total_batches: int = 1,
) -> list:
    page_payload = [page_output.model_dump(mode="json") for page_output in page_outputs]
    previous_payload = None if previous_topic_index is None else previous_topic_index.model_dump(mode="json")
    human_text = f"""Source PDF: {source_pdf_path}
Batch number: {batch_number} of {total_batches}

Previous topic index JSON:
```json
{json.dumps(previous_payload, indent=2)}
```

Latest framework memory Markdown:
```markdown
{framework_memory_md}
```

Current batch page outputs JSON:
```json
{json.dumps(page_payload, indent=2)}
```

Update the topic_index.json navigation map using only the previous topic index, latest framework memory, and current batch page outputs."""

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ]
