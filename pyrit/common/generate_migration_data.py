# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scan PyRIT's deprecation shims and validate/generate migration data.

Walks all Python files under ``pyrit/``, finds ``print_deprecation_message``
calls inside ``__getattr__`` functions, and extracts import-move rules.

Usage::

    # Validate: check that _migration_data.py covers all detected shims
    python -m pyrit.common.generate_migration_data

    # Regenerate: overwrite the generated sections of _migration_data.py
    python -m pyrit.common.generate_migration_data --write

The scanner handles the two deprecation patterns used in PyRIT:

1. **Dict-based** — ``_DEPRECATED_RENAME_ALIASES`` or ``_MOVED_TO_*`` dicts
   looked up inside ``__getattr__``, with ``print_deprecation_message``.

2. **If-chain** — ``if name == "OldName": ... print_deprecation_message(...)``
   with inline ``from X import Y`` to resolve the new location.
"""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedRule:
    """A migration rule detected from a deprecation shim."""

    old_module: str
    old_name: str
    new_module: str
    new_name: str
    removed_in: str
    source_file: str


def _module_path_from_file(*, file_path: Path, pyrit_root: Path) -> str:
    """Convert a file path to a dotted module path."""
    rel = file_path.relative_to(pyrit_root.parent)
    parts = rel.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _extract_string(node: ast.expr) -> str | None:
    """Extract a string value from an AST node (Constant or JoinedStr)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_fstring_parts(node: ast.JoinedStr) -> str | None:
    """Try to reconstruct a simple f-string like f'{__name__}.{name}'."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.Name):
            parts.append(f"{{{value.id}}}")
        else:
            return None
    return "".join(parts)


def _find_deprecation_calls_in_getattr(
    tree: ast.Module,
) -> list[ast.Call]:
    """Find all print_deprecation_message calls inside __getattr__ functions."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "print_deprecation_message"
                ):
                    calls.append(child)
    return calls


def _resolve_new_item_from_import(
    tree: ast.Module,
    name: str,
) -> tuple[str, str] | None:
    """Look for 'from X import Y' near the deprecation call to resolve new_item.

    In if-chain patterns, the new class is imported inline:
        if name == "ConsoleScorerPrinter":
            from pyrit.output.scorer.pretty import PrettyScorerMemoryPrinter
            print_deprecation_message(..., new_item=PrettyScorerMemoryPrinter, ...)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                actual_name = alias.asname or alias.name
                if actual_name == name:
                    return node.module, alias.name
    return None


def _extract_dict_mappings(
    tree: ast.Module,
    dict_name: str,
) -> dict[str, str]:
    """Extract simple {str: str} or {str: Name} dicts from module-level assignments."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    if isinstance(node.value, ast.Dict):
                        result = {}
                        for key, value in zip(node.value.keys, node.value.values):
                            k = _extract_string(key) if key else None
                            if k is None:
                                continue
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                result[k] = value.value
                            elif isinstance(value, ast.Name):
                                result[k] = value.id
                        return result
    return {}


def _parse_deprecation_call(
    call: ast.Call,
    module_path: str,
    tree: ast.Module,
) -> DetectedRule | None:
    """Parse a single print_deprecation_message call into a DetectedRule."""
    kwargs: dict[str, ast.expr] = {}
    for kw in call.keywords:
        if kw.arg:
            kwargs[kw.arg] = kw.value

    # Extract removed_in
    removed_in_node = kwargs.get("removed_in")
    removed_in = _extract_string(removed_in_node) if removed_in_node else None
    if not removed_in:
        return None

    # Extract old_item
    old_item_node = kwargs.get("old_item")
    if not old_item_node:
        return None

    old_item = _extract_string(old_item_node)
    if old_item is None and isinstance(old_item_node, ast.JoinedStr):
        fstr = _extract_fstring_parts(old_item_node)
        if fstr and "{__name__}" in fstr and "{name}" in fstr:
            # Pattern: f"{__name__}.{name}" — old_module is the module itself,
            # old_name comes from the surrounding if-check
            old_item = f"{module_path}.{{name}}"

    if not old_item:
        return None

    # Extract new_item
    new_item_node = kwargs.get("new_item")
    if not new_item_node:
        return None

    new_module: str | None = None
    new_name: str | None = None

    new_item_str = _extract_string(new_item_node)
    if new_item_str and "." in new_item_str:
        # Explicit string like "pyrit.memory.storage.storage.DiskStorageIO"
        parts = new_item_str.rsplit(".", 1)
        new_module, new_name = parts[0], parts[1]
    elif isinstance(new_item_node, ast.JoinedStr):
        fstr = _extract_fstring_parts(new_item_node)
        if fstr:
            # Pattern: f"{target_module}.{name}" — resolve target_module from dict
            pass  # handled by dict-based extraction below
    elif isinstance(new_item_node, ast.Name):
        # Reference to a class name — resolve via imports
        resolved = _resolve_new_item_from_import(tree, new_item_node.id)
        if resolved:
            new_module, new_name = resolved

    if not new_module or not new_name:
        return None

    # Parse old_item into old_module.old_name
    if "{name}" in old_item:
        old_module = old_item.replace(".{name}", "")
        old_name = new_name  # placeholder — caller fills from context
        return None  # Can't determine old_name from f-string alone
    elif "." in old_item:
        parts = old_item.rsplit(".", 1)
        old_module, old_name = parts[0], parts[1]
    else:
        return None

    return DetectedRule(
        old_module=old_module,
        old_name=old_name,
        new_module=new_module,
        new_name=new_name,
        removed_in=removed_in,
        source_file=module_path,
    )


def _scan_if_chain_pattern(
    tree: ast.Module,
    module_path: str,
) -> list[DetectedRule]:
    """Extract rules from if-chain deprecation patterns.

    Pattern:
        def __getattr__(name):
            if name == "OldName":
                from new.module import NewName
                print_deprecation_message(old_item=..., new_item=NewName, ...)
                return NewName
    """
    rules: list[DetectedRule] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "__getattr__"):
            continue

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue

            # Check: if name == "SomeName"
            old_name = _extract_if_name_check(child)
            if not old_name:
                continue

            # Find print_deprecation_message in this if block
            for stmt in ast.walk(child):
                if not (
                    isinstance(stmt, ast.Call)
                    and isinstance(stmt.func, ast.Name)
                    and stmt.func.id == "print_deprecation_message"
                ):
                    continue

                kwargs: dict[str, ast.expr] = {}
                for kw in stmt.keywords:
                    if kw.arg:
                        kwargs[kw.arg] = kw.value

                removed_in_node = kwargs.get("removed_in")
                removed_in = _extract_string(removed_in_node) if removed_in_node else None
                if not removed_in:
                    continue

                new_item_node = kwargs.get("new_item")
                if not new_item_node:
                    continue

                new_module: str | None = None
                new_name: str | None = None

                if isinstance(new_item_node, ast.Name):
                    # Resolve from inline import in the if block
                    for if_stmt in ast.walk(child):
                        if isinstance(if_stmt, ast.ImportFrom) and if_stmt.module:
                            for alias in if_stmt.names:
                                if (alias.asname or alias.name) == new_item_node.id:
                                    new_module = if_stmt.module
                                    new_name = alias.name
                                    break

                if new_module and new_name:
                    rules.append(DetectedRule(
                        old_module=module_path,
                        old_name=old_name,
                        new_module=new_module,
                        new_name=new_name,
                        removed_in=removed_in,
                        source_file=module_path,
                    ))

    return rules


def _scan_dict_pattern(
    tree: ast.Module,
    module_path: str,
) -> list[DetectedRule]:
    """Extract rules from dict-based deprecation patterns.

    Pattern A (rename aliases):
        _DEPRECATED_RENAME_ALIASES = {"ScorerIdentifier": ComponentIdentifier}

    Pattern B (module moves):
        _MOVED_TO_MEMORY_STORAGE = {"DiskStorageIO": "pyrit.memory.storage.storage", ...}
    """
    rules: list[DetectedRule] = []

    # Find all dict assignments that look like deprecation mappings
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            name = target.id
            if not (name.startswith("_DEPRECATED") or name.startswith("_MOVED")):
                continue

            if not isinstance(node.value, ast.Dict):
                continue

            # Find the removed_in version from associated print_deprecation_message calls
            removed_in = _find_removed_in_for_dict(tree, name)

            for key, value in zip(node.value.keys, node.value.values):
                old_name = _extract_string(key) if key else None
                if not old_name:
                    continue

                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    # Pattern B: value is target module path, name stays the same
                    new_module = value.value
                    rules.append(DetectedRule(
                        old_module=module_path,
                        old_name=old_name,
                        new_module=new_module,
                        new_name=old_name,
                        removed_in=removed_in or "unknown",
                        source_file=module_path,
                    ))
                elif isinstance(value, ast.Name):
                    # Pattern A: value is a class reference — resolve via imports
                    resolved = _resolve_new_item_from_import(tree, value.id)
                    if resolved:
                        new_mod, new_nm = resolved
                        rules.append(DetectedRule(
                            old_module=module_path,
                            old_name=old_name,
                            new_module=new_mod,
                            new_name=new_nm,
                            removed_in=removed_in or "unknown",
                            source_file=module_path,
                        ))
                    else:
                        # Class is defined in the same module
                        rules.append(DetectedRule(
                            old_module=module_path,
                            old_name=old_name,
                            new_module=module_path,
                            new_name=value.id,
                            removed_in=removed_in or "unknown",
                            source_file=module_path,
                        ))

    return rules


def _extract_if_name_check(if_node: ast.If) -> str | None:
    """Extract the name from ``if name == 'SomeName':``."""
    test = if_node.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], ast.Eq):
            if isinstance(test.left, ast.Name) and test.left.id == "name":
                if test.comparators and isinstance(test.comparators[0], ast.Constant):
                    val = test.comparators[0].value
                    if isinstance(val, str):
                        return val
    return None


def _find_removed_in_for_dict(tree: ast.Module, dict_name: str) -> str | None:
    """Find the removed_in version associated with a deprecation dict.

    Looks for print_deprecation_message calls that reference the dict
    (via ``name in <dict_name>`` checks in __getattr__).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "print_deprecation_message"
                ):
                    for kw in child.keywords:
                        if kw.arg == "removed_in":
                            val = _extract_string(kw.value)
                            if val:
                                return val
    return None


def scan_pyrit_deprecation_shims(pyrit_root: Path) -> list[DetectedRule]:
    """Scan all Python files under pyrit_root for deprecation shims.

    Args:
        pyrit_root: Path to the ``pyrit/`` package directory.

    Returns:
        List of detected migration rules.
    """
    all_rules: list[DetectedRule] = []
    seen: set[tuple[str, str, str, str]] = set()

    for py_file in sorted(pyrit_root.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if "print_deprecation_message" not in source:
            continue

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            logger.warning("Skipping %s (syntax error)", py_file)
            continue

        module_path = _module_path_from_file(file_path=py_file, pyrit_root=pyrit_root)

        # Try both extraction patterns
        rules = _scan_if_chain_pattern(tree, module_path)
        rules.extend(_scan_dict_pattern(tree, module_path))

        for rule in rules:
            key = (rule.old_module, rule.old_name, rule.new_module, rule.new_name)
            if key not in seen:
                seen.add(key)
                all_rules.append(rule)

    return all_rules


def validate_migration_data(pyrit_root: Path) -> list[str]:
    """Check that _migration_data.py covers all detected deprecation shims.

    Returns:
        List of warning messages for uncovered shims. Empty if all covered.
    """
    from pyrit.common._migration_data import IMPORT_RENAMES

    detected = scan_pyrit_deprecation_shims(pyrit_root)
    covered = {
        (r.old_module, r.old_name, r.new_module, r.new_name)
        for r in IMPORT_RENAMES
    }

    warnings: list[str] = []
    for rule in detected:
        key = (rule.old_module, rule.old_name, rule.new_module, rule.new_name)
        if key not in covered:
            warnings.append(
                f"Uncovered shim: {rule.old_module}.{rule.old_name} → "
                f"{rule.new_module}.{rule.new_name} (removed_in={rule.removed_in}, "
                f"source={rule.source_file})"
            )

    return warnings


def main() -> int:
    """CLI entry point for scanning and validating migration data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan PyRIT deprecation shims and validate migration data."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Print detected rules (for manual copy into _migration_data.py).",
    )
    parser.add_argument(
        "--pyrit-root",
        type=Path,
        default=None,
        help="Path to the pyrit/ package directory. Auto-detected if not set.",
    )
    args = parser.parse_args()

    if args.pyrit_root:
        pyrit_root = args.pyrit_root
    else:
        pyrit_root = Path(__file__).resolve().parent.parent

    if not (pyrit_root / "__init__.py").exists():
        print(f"Error: {pyrit_root} does not look like the pyrit package root.")
        return 1

    detected = scan_pyrit_deprecation_shims(pyrit_root)
    print(f"Detected {len(detected)} deprecation shim(s):\n")
    for rule in detected:
        print(f"  {rule.old_module}.{rule.old_name} → {rule.new_module}.{rule.new_name}  (removed_in={rule.removed_in})")

    if args.write:
        print("\n# Generated ImportRename entries:")
        for rule in detected:
            print(
                f'    ImportRename("{rule.old_module}", "{rule.old_name}", '
                f'"{rule.new_module}", "{rule.new_name}", '
                f'removed_in="{rule.removed_in}", source="shim"),'
            )

    # Validate
    print("\nValidating against _migration_data.py...")
    warnings = validate_migration_data(pyrit_root)
    if warnings:
        print(f"\n⚠️  {len(warnings)} uncovered shim(s):")
        for w in warnings:
            print(f"  {w}")
        return 1
    else:
        print("✅ All detected shims are covered in _migration_data.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
