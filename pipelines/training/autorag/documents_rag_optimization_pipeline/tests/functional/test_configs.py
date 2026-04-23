"""Test configurations for parametrized functional tests of the Documents RAG Optimization pipeline.

Configurations are loaded from test_configs.json in this directory. Each entry
specifies pipeline parameter overrides, expected result (pass/fail), and optional
tags for filtering. Use FUNCTIONAL_TESTS_TAGS (comma-separated) to run only
configs that have all of the given tags.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIGS_JSON_PATH = Path(__file__).parent / "test_configs.json"

# Milvus provider ID resolution: maps sentinel values in JSON to env var keys
# in the functional config dict.
_VECTOR_IO_PROVIDER_MAP = {
    "milvus-lite": "vector_io_provider_milvus_lite",
    "milvus-remote": "vector_io_provider_milvus_remote",
}


@dataclass
class TestConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "positive"]).
            Use FUNCTIONAL_TESTS_TAGS to run only configs that have all of the given tags.
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        pipeline_params_overrides: Keys matching pipeline parameter names. Values
            are resolved against the base config using these rules:
            - null/None: use base config value from env
            - "": pass empty string explicitly
            - "ENV": read from dedicated env var (for model lists)
            - "milvus-lite"/"milvus-remote": read provider ID from corresponding env var
            - any other value: use as-is
    """

    __test__ = False  # prevent pytest collection

    id: str
    description: str
    tags: list[str]
    expected_result: str
    llama_stack_vector_io_provider_type: str | None = None
    pipeline_params_overrides: dict[str, Any] = field(default_factory=dict)

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Build pipeline arguments dict by merging base config with overrides.

        Args:
            base_config: Functional config dict from get_functional_config().

        Returns:
            Pipeline arguments dict ready for KFP submission.
        """
        arguments: dict[str, Any] = {
            "test_data_secret_name": base_config["test_data_secret_name"],
            "test_data_bucket_name": base_config["test_data_bucket_name"],
            "test_data_key": base_config["test_data_key"],
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "llama_stack_secret_name": base_config["llama_stack_secret_name"],
        }

        overrides = self.pipeline_params_overrides

        idk = overrides.get("input_data_key")
        arguments["input_data_key"] = idk if idk is not None else base_config["input_data_key"]

        if self.llama_stack_vector_io_provider_type is None:
            arguments["llama_stack_vector_io_provider_id"] = ""
        else:
            config_key = _VECTOR_IO_PROVIDER_MAP.get(self.llama_stack_vector_io_provider_type)
            arguments["llama_stack_vector_io_provider_id"] = base_config.get(config_key, "")

        for key in ("optimization_metric", "optimization_max_rag_patterns"):
            val = overrides.get(key)
            if val is not None:
                arguments[key] = val

        for key in ("embeddings_models", "generation_models"):
            val = overrides.get(key)
            if val == "ENV":
                env_val = base_config.get(key)
                if env_val is not None:
                    arguments[key] = env_val
            elif val is not None:
                arguments[key] = val

        return arguments


def _load_configs(pass_type: str) -> list[TestConfig]:
    """Load test configs from test_configs.json and return TestConfig instances."""
    with open(_CONFIGS_JSON_PATH) as f:
        all_items = json.load(f)

    expected = "pass" if pass_type == "positive" else "fail"
    data = [item for item in all_items if item.get("expected_result") == expected]

    configs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"test_configs[{i}] must be a dict; got {type(item).__name__}")
        try:
            raw_tags = item.get("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(f"test_configs[{i}] 'tags' must be a list; got {type(raw_tags).__name__}")

            expected_result = item["expected_result"]
            if expected_result not in ("pass", "fail"):
                raise ValueError(
                    f"test_configs[{i}] 'expected_result' must be 'pass' or 'fail'; got '{expected_result}'"
                )

            configs.append(
                TestConfig(
                    id=item["id"],
                    description=item.get("description", ""),
                    tags=tags,
                    expected_result=expected_result,
                    llama_stack_vector_io_provider_type=item.get("llama_stack_vector_io_provider_type"),
                    pipeline_params_overrides=item.get("pipeline_params_overrides") or {},
                )
            )
        except KeyError as e:
            raise ValueError(f"test_configs[{i}] missing required key {e}") from e
    return configs


def get_test_configs_for_run(pass_type: str, tags: None | list[str] = None) -> list[TestConfig]:
    """Return configs to run for this session, optionally filtered by tags.

    If tags are passed, only configs that have all of those tags are returned.
    All configs are returned otherwise.

    Args:
        pass_type (str): Type of pass to run for this session. 'positive' or negative'
        tags (None | list[str]): List of tags to run for this session.

    Returns:
        list[TestConfig]: List of TestConfig instances.
    """
    test_configs: list[TestConfig] = _load_configs(pass_type)

    tags = tags or []

    env_tags_raw = os.getenv("FUNCTIONAL_TESTS_TAGS")
    env_tags = [t.strip().lower() for t in env_tags_raw.split(",") if t.strip()] if env_tags_raw else []

    all_tags = {t.lower() for t in (tags + env_tags)}

    if not all_tags:
        return test_configs
    return [c for c in test_configs if all(t.lower() in c.tags for t in all_tags)]
