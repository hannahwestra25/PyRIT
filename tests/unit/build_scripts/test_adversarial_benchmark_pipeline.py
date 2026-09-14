# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_PATH = REPO_ROOT / ".azuredevops" / "adversarial-benchmark.yml"


def _load_pipeline() -> dict:
    return yaml.safe_load(PIPELINE_PATH.read_text(encoding="utf-8"))


def test_benchmark_cache_is_enabled_and_passed_to_scenario() -> None:
    pipeline = _load_pipeline()
    parameters = {parameter["name"]: parameter for parameter in pipeline["parameters"]}
    run_step = next(
        step
        for step in pipeline["jobs"][0]["steps"]
        if step.get("displayName") == "Run benchmark and capture result snapshot"
    )

    assert parameters["useCached"]["default"] is True
    assert '--use-cached "$USE_CACHED_INPUT"' in run_step["inputs"]["inlineScript"]
    assert run_step["env"]["USE_CACHED_INPUT"] == "${{ parameters.useCached }}"


def test_benchmark_cache_restores_same_branch_state_including_failed_runs() -> None:
    pipeline = _load_pipeline()
    steps = pipeline["jobs"][0]["steps"]
    conditional_steps = next(step for step in steps if "${{ if eq(parameters.useCached, true) }}" in step)
    restore = conditional_steps["${{ if eq(parameters.useCached, true) }}"][0]

    assert restore["task"] == "DownloadPipelineArtifact@2"
    assert restore["inputs"]["buildVersionToDownload"] == "latestFromBranch"
    assert restore["inputs"]["branchName"] == "$(Build.SourceBranch)"
    assert restore["inputs"]["allowPartiallySucceededBuilds"] is True
    assert restore["inputs"]["allowFailedBuilds"] is True
    assert restore["inputs"]["artifactName"] == "adversarial-benchmark-db"
    assert restore["inputs"]["targetPath"] == "$(Build.SourcesDirectory)/dbdata"


def test_benchmark_database_is_published_even_after_failure() -> None:
    pipeline = _load_pipeline()
    steps = pipeline["jobs"][0]["steps"]
    stage = next(step for step in steps if step.get("displayName") == "Stage reusable benchmark database")
    publish = next(step for step in steps if step.get("displayName") == "Publish reusable benchmark database")

    assert stage["condition"] == "always()"
    assert "##vso[task.setvariable variable=hasBenchmarkDatabase]true" in stage["bash"]
    assert "##vso[task.setvariable variable=hasBenchmarkDatabase]false" in stage["bash"]
    assert publish["task"] == "PublishPipelineArtifact@1"
    assert publish["condition"] == "and(always(), eq(variables['hasBenchmarkDatabase'], 'true'))"
    assert publish["inputs"]["artifactName"] == "adversarial-benchmark-db"
