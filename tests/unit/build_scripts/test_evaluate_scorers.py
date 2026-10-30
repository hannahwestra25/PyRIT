# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from unit.mocks import MockPromptTarget

from build_scripts import evaluate_scorers
from pyrit.exceptions.exception_classes import handle_bad_request_exception
from pyrit.memory import SQLiteMemory
from pyrit.models import Message, MessagePiece
from pyrit.registry import ScorerRegistry
from pyrit.score import (
    ScorerEvalDatasetFiles,
    ScorerMetrics,
    SelfAskRefusalScorer,
    SubStringScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
)
from pyrit.setup import IN_MEMORY
from pyrit.setup.initializers import ScorerInitializer, TargetInitializer
from pyrit.setup.initializers.targets import TargetInitializerTags


async def test_evaluate_scorers_preserves_initializer_defaults_async() -> None:
    with (
        patch.object(evaluate_scorers, "initialize_pyrit_async", new_callable=AsyncMock) as initialize_mock,
        patch.object(ScorerRegistry, "get_registry_singleton", return_value=ScorerRegistry()),
    ):
        await evaluate_scorers.evaluate_scorers()

    initialize_mock.assert_awaited_once()
    assert initialize_mock.await_args is not None
    assert initialize_mock.await_args.kwargs["memory_db_type"] == IN_MEMORY
    target_init, scorer_init = initialize_mock.await_args.kwargs["initializers"]
    assert isinstance(target_init, TargetInitializer)
    target_init.validate_params()
    assert target_init.params["tags"] == [TargetInitializerTags.DEFAULT, TargetInitializerTags.SCORER]
    assert target_init.params["auto_group"] is True
    assert isinstance(scorer_init, ScorerInitializer)
    scorer_init.validate_params()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("blocked_count", [0, 1, 2])
async def test_evaluate_scorers_handles_blocked_composite_judgments_async(
    *, blocked_count: int, tmp_path: Path, sqlite_instance: SQLiteMemory, capsys: pytest.CaptureFixture[str]
) -> None:
    await asyncio.to_thread(
        (tmp_path / "synthetic.csv").write_text,
        "# dataset_version=1.0\nassistant_response,objective,human_score\n"
        "marker first,Synthetic objective,1\nmarker second,Synthetic objective,1\n",
        encoding="utf-8",
    )
    target = MockPromptTarget()
    refusal = SelfAskRefusalScorer(chat_target=target)
    scorer = TrueFalseCompositeScorer(
        aggregator=TrueFalseScoreAggregator.AND,
        scorers=[SubStringScorer(substring="marker"), TrueFalseInverterScorer(scorer=refusal)],
    )
    scorer.evaluation_file_mapping = ScorerEvalDatasetFiles(
        human_labeled_datasets_files=["synthetic.csv"], result_file="metrics.jsonl"
    )
    registry = ScorerRegistry()
    registry.instances.register(scorer, name="composite", tags=["objective_composite"])

    async def send_async(*, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]
        if blocked_count == 2 or (blocked_count == 1 and "marker first" in request.converted_value):
            return [
                handle_bad_request_exception(
                    response_text='{"error": {"code": "cyber_policy", "message": "Synthetic policy block"}}',
                    request=request,
                )
            ]
        return [
            MessagePiece(
                role="assistant",
                original_value='{"score_value": "False", "description": "Not a refusal", "rationale": "Synthetic"}',
                conversation_id=request.conversation_id,
            ).to_message()
        ]

    with (
        patch.object(evaluate_scorers, "initialize_pyrit_async", new_callable=AsyncMock),
        patch.object(ScorerRegistry, "get_registry_singleton", return_value=registry),
        patch("pyrit.score.scorer_evaluation.scorer_evaluator.SCORER_EVALS_PATH", tmp_path),
        patch.object(target, "_send_prompt_to_target_async", new_callable=AsyncMock, side_effect=send_async) as send,
    ):
        if blocked_count == 2:
            with pytest.raises(RuntimeError, match="Evaluation failed for 1 scorer"):
                await evaluate_scorers.evaluate_scorers(tags=["objective_composite"])
        else:
            await evaluate_scorers.evaluate_scorers(tags=["objective_composite"])

    assert send.await_count == 6
    assert refusal.raise_if_scorer_blocks is True
    assert registry.instances.get("composite") is scorer
    scores = await sqlite_instance.get_scores_async(score_type="true_false", include_intermediate=True)
    judge_scores = [score for score in scores if score.scorer_class_identifier.class_name == "SelfAskRefusalScorer"]
    assert len(judge_scores) == 6
    assert sum(score.is_undetermined for score in judge_scores) == blocked_count * 3
    for score in judge_scores:
        if score.is_undetermined:
            assert score.score_value is None
            assert score.observation_ids
            assert "blocked by content filtering" in score.score_rationale

    output = capsys.readouterr().out
    result_path = tmp_path / "metrics.jsonl"
    if blocked_count == 2:
        assert not result_path.exists()
        assert "no determined scores" in output
        assert "Evaluation complete" not in output
    else:
        record = json.loads(await asyncio.to_thread(result_path.read_text, encoding="utf-8"))
        assert record["metrics"]["num_input_responses"] == 2
        assert record["metrics"]["num_responses"] == 2 - blocked_count
        assert record["metrics"]["num_scorer_trials"] == 3
        assert record["metrics"]["accuracy"] == 1.0
        assert f"Coverage: {2 - blocked_count}/2 responses evaluated." in output
        assert ("WARNING: Partial coverage" in output) == (blocked_count == 1)


@pytest.mark.usefixtures("patch_central_database")
async def test_evaluate_scorers_reports_unknown_legacy_coverage_async(capsys: pytest.CaptureFixture[str]) -> None:
    scorer = SubStringScorer(substring="marker")
    registry = ScorerRegistry()
    registry.instances.register(scorer, name="legacy")
    metrics = ScorerMetrics(num_responses=3, num_human_raters=1, dataset_name="legacy.csv")
    with (
        patch.object(evaluate_scorers, "initialize_pyrit_async", new_callable=AsyncMock),
        patch.object(ScorerRegistry, "get_registry_singleton", return_value=registry),
        patch.object(scorer, "evaluate_async", new_callable=AsyncMock, return_value=metrics),
    ):
        await evaluate_scorers.evaluate_scorers()

    assert "Coverage: 3 evaluated; input count unavailable in legacy metrics." in capsys.readouterr().out


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("error", [None, ValueError("Invalid dataset")])
async def test_evaluate_scorers_reports_failures_after_remaining_scorers_async(
    *, error: ValueError | None, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = SubStringScorer(substring="failing")
    succeeding = SubStringScorer(substring="succeeding")
    registry = ScorerRegistry()
    registry.instances.register(failing, name="a_failing")
    registry.instances.register(succeeding, name="b_succeeding")
    metrics = ScorerMetrics(num_responses=1, num_input_responses=1, num_human_raters=1)
    with (
        patch.object(evaluate_scorers, "initialize_pyrit_async", new_callable=AsyncMock),
        patch.object(ScorerRegistry, "get_registry_singleton", return_value=registry),
        patch.object(failing, "evaluate_async", new_callable=AsyncMock, return_value=None, side_effect=error),
        patch.object(succeeding, "evaluate_async", new_callable=AsyncMock, return_value=metrics) as succeeding_eval,
    ):
        with pytest.raises(RuntimeError, match="Evaluation failed for 1 scorer\\(s\\): a_failing"):
            await evaluate_scorers.evaluate_scorers()

    succeeding_eval.assert_awaited_once()
    output = capsys.readouterr().out
    assert ("Invalid dataset" if error else "Evaluation returned no metrics") in output
    assert "Coverage: 1/1 responses evaluated." in output
