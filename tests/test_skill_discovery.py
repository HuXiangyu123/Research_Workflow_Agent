from __future__ import annotations

from pathlib import Path

from src.skills.discovery import SkillMetadataParser


def test_load_skill_content_reads_skill_markdown_body():
    skill_dir = Path(".agents/skills/writing_scaffold_generator")
    content = SkillMetadataParser.load_skill_content(skill_dir)

    assert "Writing Scaffold Generator" in content
    assert "Use this skill immediately before drafting" in content

