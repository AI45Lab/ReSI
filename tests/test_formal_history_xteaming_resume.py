from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from r2ai_autoresearch.formal_eval.config import Endpoint, GenerationConfig
from r2ai_autoresearch.formal_eval.xteaming import (
    Role,
    _canonical_behavior_records,
    _DirectTargetSession,
    _record_is_protocol_complete,
    _target_output_violations,
)


class XTeamingResumeTests(unittest.TestCase):
    def test_incomplete_planning_does_not_start_attack_execution(self):
        from r2ai_autoresearch.formal_eval.xteaming import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text(json.dumps({"roles": {"target": {}}}))
            with patch("r2ai_autoresearch.formal_eval.xteaming._load_protocol_prompts"), \
                 patch("r2ai_autoresearch.formal_eval.xteaming._resolved_config", return_value={}), \
                 patch("r2ai_autoresearch.formal_eval.xteaming.load_protocol_behaviors", return_value=[{"id": "fixture"}]), \
                 patch("r2ai_autoresearch.formal_eval.xteaming.PlanGenerator") as planner, \
                 patch("r2ai_autoresearch.formal_eval.xteaming.AttackRunner") as runner:
                planner.return_value.run.return_value = {"complete_behaviors": 0, "errors_this_run": 1}
                code = main(["--config", str(config), "--plans-dir", str(root / "plans"), "--stage", "all"])
            self.assertEqual(code, 2)
            runner.assert_not_called()

    def test_planning_count_matches_prompt_and_rejects_impossible_selection(self):
        from r2ai_autoresearch.formal_eval.xteaming import PlanGenerator, _load_protocol_prompts
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"roles": {"planner": {"model": "fixture", "base_url": "http://localhost/v1"}},
                      "planner_sets": 1, "plans_per_set": 1, "plans_to_execute": 1}
            prompts = _load_protocol_prompts(config, root / "config.json")
            strategy = {"approach": "fixture", "conversation_plan": {"final_turn": "fixture"}}
            with patch("r2ai_autoresearch.formal_eval.xteaming._call", return_value={"response": json.dumps({"strategy_1": strategy})}) as call, \
                 patch("r2ai_autoresearch.formal_eval.xteaming._validate_strategy", side_effect=lambda item: item):
                generator = PlanGenerator(config, root / "config.json", prompts, root / "plans")
                result = generator.generate_one({"id": "fixture", "prompt": "fixture"})
            self.assertEqual(result["status"], "complete")
            self.assertIn("Generate 1 distinct jailbreak strategies", call.call_args.args[1])
            config["plans_to_execute"] = 2
            with self.assertRaisesRegex(ValueError, "exceeds"):
                PlanGenerator(config, root / "config.json", prompts, root / "plans")

    def test_valid_finished_records_are_reused(self):
        self.assertTrue(
            _record_is_protocol_complete(
                {"status": "success", "reasoning_violation_count": 0}
            )
        )
        self.assertTrue(_record_is_protocol_complete({"status": "exhausted"}))

    def test_reasoning_violations_are_retried(self):
        self.assertFalse(
            _record_is_protocol_complete(
                {"status": "success", "reasoning_violation_count": 1}
            )
        )
        self.assertFalse(
            _record_is_protocol_complete(
                {"status": "exhausted", "reasoning_violation_count": 2}
            )
        )

    def test_valid_success_supersedes_unscoreable_alternative_plan_error(self):
        records = [
            {"plan_index": 0, "status": "error", "success": False},
            {"plan_index": 1, "status": "success", "success": True},
        ]
        canonical = _canonical_behavior_records(records, 10)
        self.assertEqual([item["plan_index"] for item in canonical], [1])

    def test_canonical_records_ignore_unreachable_error_after_success(self):
        records = [
            {"plan_index": 0, "status": "success", "success": True},
            {"plan_index": 1, "status": "error", "success": False},
        ]
        canonical = _canonical_behavior_records(records, 10)
        self.assertEqual([item["plan_index"] for item in canonical], [0])

    def test_canonical_records_preserve_exhausted_prefix(self):
        records = [
            {"plan_index": 0, "status": "exhausted", "success": False},
            {"plan_index": 1, "status": "error", "success": False},
            {"plan_index": 2, "status": "success", "success": True},
        ]
        canonical = _canonical_behavior_records(records, 10)
        self.assertEqual([item["plan_index"] for item in canonical], [0, 2])

    def test_without_valid_success_all_latest_errors_remain_active(self):
        records = [
            {"plan_index": 0, "status": "exhausted", "success": False},
            {"plan_index": 1, "status": "error", "success": False},
            {"plan_index": 2, "status": "exhausted", "success": False},
        ]
        canonical = _canonical_behavior_records(records, 10)
        self.assertEqual([item["plan_index"] for item in canonical], [0, 1, 2])


class DirectTargetSessionTests(unittest.TestCase):
    @staticmethod
    def _role() -> Role:
        return Role(
            endpoint=Endpoint(model="target", base_url="http://target/v1"),
            generation=GenerationConfig(seed=42),
        )

    @patch("r2ai_autoresearch.formal_eval.xteaming._call_messages")
    def test_invalid_output_is_retried_without_entering_history(self, call):
        call.side_effect = [
            {
                "response": "",
                "reasoning_content": "first reasoning",
                "finish_reason": "stop",
            },
            {
                "response": "valid final",
                "reasoning_content": "second reasoning",
                "finish_reason": "stop",
            },
        ]
        session = _DirectTargetSession(
            self._role(), reasoning_policy="required", semantic_retries=5
        )

        response, result = session.generate("question")

        self.assertEqual(response, "valid final")
        self.assertEqual(result["target_output_retry_count"], 1)
        self.assertEqual(
            result["target_output_retries"][0]["reasoning_violations"],
            ["empty_target_response"],
        )
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "valid final"},
            ],
        )
        self.assertEqual(call.call_args_list[0].args[0].generation.seed, 42)
        self.assertEqual(call.call_args_list[1].args[0].generation.seed, 43)

    @patch("r2ai_autoresearch.formal_eval.xteaming._call_messages")
    def test_exhausted_invalid_outputs_never_enter_history(self, call):
        call.side_effect = [
            {
                "response": "",
                "reasoning_content": f"reasoning {index}",
                "finish_reason": "stop",
            }
            for index in range(5)
        ]
        session = _DirectTargetSession(
            self._role(), reasoning_policy="required", semantic_retries=5
        )

        response, result = session.generate("question")

        self.assertEqual(response, "")
        self.assertEqual(result["target_output_retry_count"], 4)
        self.assertEqual(len(result["target_output_retries"]), 4)
        self.assertEqual(session.messages, [])
        self.assertEqual(call.call_count, 5)

    def test_answer_only_accepts_nonempty_final_without_reasoning(self):
        self.assertEqual(
            _target_output_violations(
                {"reasoning_content": None}, "final answer", "answer_only"
            ),
            [],
        )

    def test_answer_only_still_rejects_empty_final(self):
        self.assertEqual(
            _target_output_violations(
                {"reasoning_content": "reasoning"}, "", "answer_only"
            ),
            ["empty_target_response"],
        )
