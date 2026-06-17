from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.constants import WORKDIR


SKILLS_DIR = WORKDIR / "skills"


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


def discover_skills() -> list[SkillInfo]:
    skills: list[SkillInfo] = []
    if not SKILLS_DIR.exists():
        return skills

    for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        metadata = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        name = metadata.get("name") or skill_file.parent.name
        description = metadata.get("description") or "(no description)"
        skills.append(SkillInfo(name=name, description=description, path=skill_file))

    return skills


def build_skill_directory_section() -> str:
    skills = discover_skills()
    if not skills:
        return (
            "Available skills:\n"
            "(none found)\n\n"
            "When a task would benefit from a skill, call load_skill with the skill name "
            "to read its full instructions before continuing."
        )

    entries = "\n".join(
        f"- {skill.name}: {skill.description}"
        for skill in skills
    )
    return (
        "Available skills:\n"
        f"{entries}\n\n"
        "When a task matches a skill description, call load_skill with that skill name "
        "to read the full SKILL.md instructions before continuing."
    )


def load_skill_content(skill_name: str) -> str:
    requested_name = str(skill_name or "").strip()
    skills_by_name = {skill.name: skill for skill in discover_skills()}
    skill = skills_by_name.get(requested_name)
    if skill is None:
        available = ", ".join(sorted(skills_by_name)) or "(none)"
        return (
            f"Skill not found: {requested_name or '(empty)'}\n"
            f"Available skills: {available}"
        )

    return skill.path.read_text(encoding="utf-8")


def _parse_frontmatter(content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break

    if end_index is None:
        return {}

    return _parse_simple_yaml(lines[1:end_index])


def _parse_simple_yaml(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            index += 1
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value in {">", ">-", "|", "|-"}:
            block_lines = []
            index += 1
            while index < len(lines) and _is_indented(lines[index]):
                block_lines.append(lines[index].strip())
                index += 1
            metadata[key] = " ".join(part for part in block_lines if part)
            continue

        metadata[key] = _strip_yaml_quotes(value)
        index += 1

    return metadata


def _is_indented(line: str) -> bool:
    return line.startswith(" ") or line.startswith("\t")


def _strip_yaml_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
