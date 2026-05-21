# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from pyrit.scenario.scenarios.adaptive.selectors import (
    GLOBAL_CONTEXT,
    UNCATEGORIZED_CONTEXT,
    EpsilonGreedyTechniqueSelector,
    TechniqueSelector,
    global_context,
    harm_category_context,
)


class TestTechniqueSelectorProtocol:
    def test_implements_protocol(self):
        selector = EpsilonGreedyTechniqueSelector()
        assert isinstance(selector, TechniqueSelector)


class TestContextExtractors:
    def test_global_context_is_constant(self):
        sg = MagicMock()
        assert global_context(sg) == GLOBAL_CONTEXT

    def test_harm_category_context_joins_sorted_categories(self):
        sg = MagicMock()
        sg.harm_categories = ["violence", "hate"]
        # Multi-category seeds form their own bucket; sorting keeps the key deterministic.
        assert harm_category_context(sg) == "hate|violence"

    def test_harm_category_context_single_category(self):
        sg = MagicMock()
        sg.harm_categories = ["violence"]
        assert harm_category_context(sg) == "violence"

    def test_harm_category_context_falls_back_when_empty(self):
        sg = MagicMock()
        sg.harm_categories = []
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT

    def test_harm_category_context_falls_back_when_none(self):
        sg = MagicMock()
        sg.harm_categories = None
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT
