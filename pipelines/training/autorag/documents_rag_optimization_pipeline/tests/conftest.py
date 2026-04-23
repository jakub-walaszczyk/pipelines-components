"""Pytest fixtures for Documents RAG Optimization pipeline tests."""

import os
import tempfile
from pathlib import Path

import pytest

from pipelines.training.autorag.documents_rag_optimization_pipeline.tests.utils import (
    _make_kfp_client,
    _make_s3_client,
    get_docrag_integration_config,
)


@pytest.fixture(scope="session")
def docrag_integration_config():
    """Session-scoped RHOAI integration config fixture."""
    return get_docrag_integration_config()


@pytest.fixture(scope="session")
def kfp_client(docrag_integration_config):
    """Session-scoped KFP client for integration tests."""
    return _make_kfp_client(docrag_integration_config)


@pytest.fixture(scope="session")
def compiled_pipeline_path():
    """Compile the Documents RAG Optimization pipeline to a temp YAML file."""
    from kfp import compiler

    from ..pipeline import documents_rag_optimization_pipeline

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    compiler.Compiler().compile(
        pipeline_func=documents_rag_optimization_pipeline,
        package_path=path,
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


@pytest.fixture(scope="session")
def s3_client(docrag_integration_config):
    """Session-scoped S3 client for integration test artifact checks (optional)."""
    return _make_s3_client(docrag_integration_config)
