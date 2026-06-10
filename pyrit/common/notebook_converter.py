# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Notebook Converter for PyRIT.

Reads a .ipynb or .py file and rewrites it to be compatible with the
latest version of PyRIT.  Changes are categorized by confidence:

- **auto**: Safe to apply automatically (import rewrites, class renames).
- **suggestion**: Likely correct but may need human review (kwarg/method
  renames that could be ambiguous).

Usage::

    from pyrit.common.notebook_converter import convert_notebook

    result = convert_notebook("my_notebook.ipynb")
    print(result.diff())           # preview changes
    result.write()                 # write the converted file

    # Or convert raw source code:
    from pyrit.common.notebook_converter import convert_source
    converted, changes = convert_source(old_code)
"""

from __future__ import annotations

import copy
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Change tracking
# ---------------------------------------------------------------------------


@dataclass
class Change:
    """A single change applied (or suggested) by the converter."""

    line_number: int
    description: str
    old_text: str
    new_text: str
    confidence: str  # "auto" or "suggestion"
    category: str  # "import", "class_rename", "kwarg", "method", "module"


@dataclass
class ConversionResult:
    """Result of converting a notebook or source file."""

    original_source: str
    converted_source: str
    changes: list[Change] = field(default_factory=list)
    file_path: str | None = None

    def has_changes(self) -> bool:
        return bool(self.changes)

    def diff(self, context_lines: int = 3) -> str:
        """Return a unified diff of the changes."""
        old_name = self.file_path or "original"
        new_name = f"{old_name} (converted)"
        return "".join(
            unified_diff(
                self.original_source.splitlines(keepends=True),
                self.converted_source.splitlines(keepends=True),
                fromfile=old_name,
                tofile=new_name,
                n=context_lines,
            )
        )

    def summary(self) -> str:
        """Return a human-readable summary of changes."""
        if not self.changes:
            return "No changes needed — notebook is up to date!"

        auto = [c for c in self.changes if c.confidence == "auto"]
        suggestions = [c for c in self.changes if c.confidence == "suggestion"]

        lines = [f"Found {len(self.changes)} change(s):"]
        if auto:
            lines.append(f"  ✅ {len(auto)} auto-applied")
        if suggestions:
            lines.append(f"  💡 {len(suggestions)} suggestion(s) — review recommended")
        lines.append("")
        for c in self.changes:
            marker = "✅" if c.confidence == "auto" else "💡"
            lines.append(f"  {marker} Line {c.line_number}: {c.description}")
        return "\n".join(lines)

    def write(self, output_path: str | None = None) -> str:
        """Write the converted source to disk.

        Args:
            output_path: Where to write. Defaults to overwriting the original.

        Returns:
            The path written to.
        """
        path = output_path or self.file_path
        if not path:
            raise ValueError("No output path specified and no original file path available.")
        Path(path).write_text(self.converted_source, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Import rewrite rules (from shared migration data)
# ---------------------------------------------------------------------------

from pyrit.common._migration_data import (
    CLASS_RENAMES as _CLASS_RENAMES_DATA,
    IMPORT_RENAMES as _IMPORT_RENAMES_DATA,
    KWARG_RENAMES as _KWARG_RENAMES_DATA,
    METHOD_RENAMES as _METHOD_RENAMES_DATA,
    MODULE_RENAMES as _MODULE_RENAMES_DATA,
)

# Unpack shared data into the formats used by the converter engine
_IMPORT_RENAMES: list[tuple[str, str, str, str]] = [
    (r.old_module, r.old_name, r.new_module, r.new_name)
    for r in _IMPORT_RENAMES_DATA
]

_MODULE_RENAMES: dict[str, str] = {
    r.old_module: r.new_module for r in _MODULE_RENAMES_DATA
}

_CLASS_RENAMES: list[tuple[str, str]] = [
    (r.old_name, r.new_name) for r in _CLASS_RENAMES_DATA
]

# Pre-compile patterns for class renames (longest first to avoid partial matches)
_CLASS_RENAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{re.escape(old)}\b"), new)
    for old, new in sorted(_CLASS_RENAMES, key=lambda x: -len(x[0]))
]

_KWARG_RENAMES: list[tuple[str | None, str, str, str]] = [
    (r.context_pattern, r.old_kwarg, r.new_kwarg, r.confidence)
    for r in _KWARG_RENAMES_DATA
]

_METHOD_RENAMES: list[tuple[str | None, str, str, str]] = [
    (r.context_pattern, r.old_method, r.new_method, r.confidence)
    for r in _METHOD_RENAMES_DATA
]

# ---------------------------------------------------------------------------
# Import parsing and rewriting
# ---------------------------------------------------------------------------

# Regex to match "from X import Y" or "from X import Y as Z"
_FROM_IMPORT_RE = re.compile(
    r"^(\s*)"                     # leading whitespace
    r"from\s+([\w.]+)\s+"         # from module
    r"import\s+"                  # import keyword
    r"(.+)"                       # imported names (may include aliases, parens)
    r"$",
    re.MULTILINE,
)

# Regex to match "import X" or "import X as Y"
_BARE_IMPORT_RE = re.compile(
    r"^(\s*)import\s+([\w.]+)(\s+as\s+(\w+))?\s*$",
    re.MULTILINE,
)


def _parse_import_names(names_str: str) -> list[tuple[str, str | None]]:
    """Parse 'A, B as C, D' into [(A, None), (B, C), (D, None)]."""
    # Strip parentheses and continuation characters
    cleaned = names_str.strip().strip("()\\\n")
    result = []
    for part in cleaned.split(","):
        part = part.strip()
        if not part:
            continue
        if " as " in part:
            name, alias = part.split(" as ", 1)
            result.append((name.strip(), alias.strip()))
        else:
            result.append((part, None))
    return result


def _build_import_line(indent: str, module: str, names: list[tuple[str, str | None]]) -> str:
    """Build a 'from module import ...' line from parsed components."""
    parts = []
    for name, alias in names:
        if alias:
            parts.append(f"{name} as {alias}")
        else:
            parts.append(name)
    names_str = ", ".join(parts)
    return f"{indent}from {module} import {names_str}"


# Build lookup: (old_module, old_name) → (new_module, new_name)
_IMPORT_LOOKUP: dict[tuple[str, str], tuple[str, str]] = {
    (old_mod, old_name): (new_mod, new_name)
    for old_mod, old_name, new_mod, new_name in _IMPORT_RENAMES
}


def _rewrite_from_import(indent: str, module: str, names_str: str) -> tuple[str, list[Change]]:
    """Rewrite a 'from X import Y, Z' statement if any names have moved."""
    names = _parse_import_names(names_str)
    changes: list[Change] = []

    # Group names by their target module after rewriting
    by_module: dict[str, list[tuple[str, str | None]]] = {}

    for name, alias in names:
        key = (module, name)
        if key in _IMPORT_LOOKUP:
            new_mod, new_name = _IMPORT_LOOKUP[key]
            changes.append(Change(
                line_number=0,  # filled in by caller
                description=f"'{module}.{name}' → '{new_mod}.{new_name}'",
                old_text=f"from {module} import {name}",
                new_text=f"from {new_mod} import {new_name}",
                confidence="auto",
                category="import",
            ))
            by_module.setdefault(new_mod, []).append((new_name, alias))
        else:
            # Check if the module itself has been renamed
            new_mod = _MODULE_RENAMES.get(module, module)
            if new_mod != module and not changes:
                changes.append(Change(
                    line_number=0,
                    description=f"Module '{module}' → '{new_mod}'",
                    old_text=f"from {module}",
                    new_text=f"from {new_mod}",
                    confidence="auto",
                    category="module",
                ))
            by_module.setdefault(new_mod, []).append((name, alias))

    if not changes:
        return f"{indent}from {module} import {names_str}", []

    # Build new import lines (may split across multiple if names moved to different modules)
    lines = []
    for mod, mod_names in sorted(by_module.items()):
        lines.append(_build_import_line(indent, mod, mod_names))

    return "\n".join(lines), changes


def _rewrite_bare_import(indent: str, module: str, alias: str | None) -> tuple[str, list[Change]]:
    """Rewrite 'import pyrit.orchestrator' → 'import pyrit.executor.attack'."""
    new_mod = _MODULE_RENAMES.get(module)
    if not new_mod:
        return "", []

    alias_part = f" as {alias}" if alias else ""
    old_line = f"{indent}import {module}{alias_part}"
    new_line = f"{indent}import {new_mod}{alias_part}"
    change = Change(
        line_number=0,
        description=f"Module '{module}' → '{new_mod}'",
        old_text=old_line,
        new_text=new_line,
        confidence="auto",
        category="module",
    )
    return new_line, [change]


# ---------------------------------------------------------------------------
# Core conversion engine
# ---------------------------------------------------------------------------


def convert_source(source: str) -> tuple[str, list[Change]]:
    """Convert PyRIT source code to the latest API.

    Args:
        source: Python source code string.

    Returns:
        Tuple of (converted_source, list_of_changes).
    """
    all_changes: list[Change] = []
    lines = source.splitlines(keepends=True)
    result_lines: list[str] = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        original_line = line
        line_changed = False

        # --- Pass 1: Import rewrites ---
        from_match = _FROM_IMPORT_RE.match(line.rstrip("\n\r"))
        if from_match:
            indent, module, names_str = from_match.groups()
            rewritten, changes = _rewrite_from_import(indent, module, names_str)
            if changes:
                for c in changes:
                    c.line_number = line_num
                all_changes.extend(changes)
                result_lines.append(rewritten + "\n")
                line_changed = True

        bare_match = _BARE_IMPORT_RE.match(line.rstrip("\n\r")) if not line_changed else None
        if bare_match and not line_changed:
            indent = bare_match.group(1)
            module = bare_match.group(2)
            alias = bare_match.group(4)
            rewritten, changes = _rewrite_bare_import(indent, module, alias)
            if changes:
                for c in changes:
                    c.line_number = line_num
                all_changes.extend(changes)
                result_lines.append(rewritten + "\n")
                line_changed = True

        if line_changed:
            continue

        # --- Pass 2: Class name renames (word-boundary aware) ---
        # Skip comment lines and string-only lines
        stripped = line.lstrip()
        if stripped.startswith("#"):
            result_lines.append(line)
            continue

        modified = line
        for pattern, new_name in _CLASS_RENAME_PATTERNS:
            if pattern.search(modified):
                old_match_text = pattern.search(modified).group(0)  # type: ignore[union-attr]
                new_modified = pattern.sub(new_name, modified)
                if new_modified != modified:
                    all_changes.append(Change(
                        line_number=line_num,
                        description=f"Renamed '{old_match_text}' → '{new_name}'",
                        old_text=modified.rstrip("\n\r"),
                        new_text=new_modified.rstrip("\n\r"),
                        confidence="auto",
                        category="class_rename",
                    ))
                    modified = new_modified

        # --- Pass 3: Kwarg renames ---
        for context_pat, old_kwarg, new_kwarg, confidence in _KWARG_RENAMES:
            kwarg_pattern = re.compile(rf"\b{re.escape(old_kwarg)}\s*=")
            if kwarg_pattern.search(modified):
                # Check context if required
                if context_pat:
                    # Look at current line and a few preceding lines for context
                    context_window = "".join(
                        result_lines[max(0, len(result_lines) - 5):]
                    ) + modified
                    if not re.search(context_pat, context_window):
                        continue

                new_modified = kwarg_pattern.sub(f"{new_kwarg}=", modified, count=1)
                if new_modified != modified:
                    all_changes.append(Change(
                        line_number=line_num,
                        description=f"Kwarg '{old_kwarg}=' → '{new_kwarg}='",
                        old_text=modified.rstrip("\n\r"),
                        new_text=new_modified.rstrip("\n\r"),
                        confidence=confidence,
                        category="kwarg",
                    ))
                    modified = new_modified

        # --- Pass 4: Method renames ---
        for context_pat, old_method, new_method, confidence in _METHOD_RENAMES:
            method_re = re.compile(re.escape(old_method) if not old_method.startswith("\\") else old_method)
            if method_re.search(modified):
                if context_pat:
                    context_window = "".join(
                        result_lines[max(0, len(result_lines) - 5):]
                    ) + modified
                    if not re.search(context_pat, context_window):
                        continue

                new_modified = method_re.sub(new_method, modified, count=1)
                if new_modified != modified:
                    all_changes.append(Change(
                        line_number=line_num,
                        description=f"Method '{old_method.strip('.')}' → '{new_method.strip('.')}'",
                        old_text=modified.rstrip("\n\r"),
                        new_text=new_modified.rstrip("\n\r"),
                        confidence=confidence,
                        category="method",
                    ))
                    modified = new_modified

        result_lines.append(modified)

    return "".join(result_lines), all_changes


# ---------------------------------------------------------------------------
# Notebook (.ipynb) support
# ---------------------------------------------------------------------------


def _convert_notebook_json(nb: dict[str, Any]) -> tuple[dict[str, Any], list[Change]]:
    """Convert all code cells in a notebook dict."""
    nb = copy.deepcopy(nb)
    all_changes: list[Change] = []
    cell_offset = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            cell_offset += len(cell.get("source", []))
            continue

        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            source = "".join(source_lines)
        else:
            source = source_lines

        converted, changes = convert_source(source)

        if changes:
            # Adjust line numbers to be relative to the notebook
            for c in changes:
                c.line_number += cell_offset
            all_changes.extend(changes)

            # Write back as list of lines (notebook format)
            cell["source"] = converted.splitlines(keepends=True)

        cell_offset += len(source.splitlines())

    return nb, all_changes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_output_path(path: Path) -> Path:
    """Generate a default output path by adding '_converted' before the extension."""
    return path.with_stem(f"{path.stem}_converted")


def convert_notebook(
    path: str | Path,
    *,
    write: bool = True,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert a .ipynb or .py file to the latest PyRIT API.

    By default, writes a converted copy next to the original (e.g.
    ``my_notebook_converted.ipynb``). The original file is never modified
    unless ``output_path`` is explicitly set to the same path.

    Args:
        path: Path to the notebook (.ipynb) or Python file (.py).
        write: If True, write the converted file. Defaults to True.
        output_path: Where to write the output. Defaults to a copy with
            ``_converted`` appended to the filename stem.

    Returns:
        A ConversionResult with the original source, converted source,
        list of changes, and a diff.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    str_path = str(path)

    if path.suffix == ".ipynb":
        raw = path.read_text(encoding="utf-8")
        nb = json.loads(raw)
        converted_nb, changes = _convert_notebook_json(nb)

        if changes:
            converted_source = json.dumps(converted_nb, indent=1, ensure_ascii=False) + "\n"
        else:
            converted_source = raw

        result = ConversionResult(
            original_source=raw,
            converted_source=converted_source,
            changes=changes,
            file_path=str_path,
        )
    else:
        original = path.read_text(encoding="utf-8")
        converted, changes = convert_source(original)
        result = ConversionResult(
            original_source=original,
            converted_source=converted,
            changes=changes,
            file_path=str_path,
        )

    if write and result.has_changes():
        if output_path:
            out = str(output_path)
        else:
            out = str(_default_output_path(path))
        result.write(out)
        logger.info("Wrote converted file to %s", out)

    return result
