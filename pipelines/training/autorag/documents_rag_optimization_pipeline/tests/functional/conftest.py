import json
import logging
import os

import pytest

from pipelines.training.autorag.documents_rag_optimization_pipeline.tests.utils import (
    FUNC_TEST_EMBEDDINGS_MODELS_ENV,
    FUNC_TEST_GENERATION_MODELS_ENV,
    LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_LITE_ENV,
    LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_REMOTE_ENV,
    build_base_env_config,
    make_kfp_client,
    make_s3_client,
)

logger = logging.getLogger(__name__)


def _parse_json_list(env_name):
    raw = os.environ.get(env_name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{env_name} must be a JSON array, got {type(value).__name__}")
    return value


def get_functional_config():
    """Build functional test config from environment; None if not configured.

    Relaxed guards compared to integration config (does not require
    llama_stack_vector_io_provider_id or input_data_key since those are
    overridden per-scenario). Adds milvus provider IDs and constrained model lists.
    """
    base = build_base_env_config()
    if base is None:
        logger.info("Missing required environmental variables. Functional config cannot be created.")
        return None

    if not base["rhoai_project"]:
        logger.info("Missing RHOAI_PROJECT_NAME. Functional config cannot be created.")
        return None

    base["vector_io_provider_milvus_lite"] = os.environ.get(LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_LITE_ENV, "").strip()
    base["vector_io_provider_milvus_remote"] = os.environ.get(
        LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_REMOTE_ENV, ""
    ).strip()
    base["embeddings_models"] = _parse_json_list(FUNC_TEST_EMBEDDINGS_MODELS_ENV)
    base["generation_models"] = _parse_json_list(FUNC_TEST_GENERATION_MODELS_ENV)
    return base


@pytest.fixture(scope="session")
def functional_env_config():
    """Session-scoped functional test config fixture."""
    return get_functional_config()


@pytest.fixture(scope="session")
def kfp_client_functional(functional_env_config):
    """Session-scoped KFP client for functional tests."""
    return make_kfp_client(functional_env_config)


@pytest.fixture(scope="session")
def s3_client_functional(functional_env_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    return make_s3_client(functional_env_config)
