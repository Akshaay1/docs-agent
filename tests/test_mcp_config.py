"""Tests for docs-agent-mcp/mcp-server/config.py and its call sites.

config.py reads numeric settings from the environment at import time. A typo
in the mcp-server-config ConfigMap used to crash the server with a bare
ValueError traceback that never named the variable. These tests cover
malformed, valid, and unset values for the helpers, and prove every call site
in server.py, milvus_search.py, and intent_router.py reads through them.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

MCP_SERVER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"

sys.modules.setdefault("pymilvus", MagicMock())
sys.path.insert(0, str(MCP_SERVER_DIR))

# Modules that read env vars in their module bodies; re-imported per test.
CONSUMER_MODULES = ("server", "milvus_search", "intent_router", "config")

ALL_ENV_VARS = (
    "PORT",
    "MAX_QUERY_CHARS",
    "MAX_TOP_K",
    "DOCS_CONTEXT_MAX_CHARS",
    "HYBRID_DENSE_WEIGHT",
    "HYBRID_SPARSE_WEIGHT",
    "HYBRID_RRF_K",
    "AUTO_TEMPORAL_CANDIDATE_DEPTH",
    "AUTO_BM25_CANDIDATE_DEPTH",
)


def _load_config():
    spec = importlib.util.spec_from_file_location("mcp_config", MCP_SERVER_DIR / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_config()


@pytest.fixture
def fresh_consumers(monkeypatch):
    """Re-import the consumer modules per test.

    server.py, milvus_search.py, and intent_router.py read env vars at import
    time, so each test gets modules that were imported after its monkeypatched
    env went into effect. syspath_prepend guards against other test files that
    push their own directories to the front of sys.path at collection time.
    """
    monkeypatch.syspath_prepend(str(MCP_SERVER_DIR))
    for name in CONSUMER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield monkeypatch
    for name in CONSUMER_MODULES:
        sys.modules.pop(name, None)


class TestEnvInt:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("PORT", raising=False)
        assert config.env_int("PORT", 8000) == 8000

    def test_valid_value_is_parsed(self, monkeypatch):
        monkeypatch.setenv("PORT", "9000")
        assert config.env_int("PORT", 8000) == 9000

    def test_surrounding_whitespace_is_accepted(self, monkeypatch):
        # int() ignores surrounding whitespace, same as the old code.
        monkeypatch.setenv("PORT", " 9000 ")
        assert config.env_int("PORT", 8000) == 9000

    @pytest.mark.parametrize("bad", ["eighty", "80o0", "8000.0", "", "0x1F90"])
    def test_malformed_value_names_variable_type_value_and_default(self, monkeypatch, bad):
        monkeypatch.setenv("PORT", bad)
        with pytest.raises(ValueError) as excinfo:
            config.env_int("PORT", 8000)
        message = str(excinfo.value)
        assert "PORT" in message
        assert "integer" in message
        assert repr(bad) in message
        assert "8000" in message

    def test_minimum_and_maximum_are_inclusive(self, monkeypatch):
        monkeypatch.setenv("PORT", "1")
        assert config.env_int("PORT", 8000, minimum=1, maximum=65535) == 1
        monkeypatch.setenv("PORT", "65535")
        assert config.env_int("PORT", 8000, minimum=1, maximum=65535) == 65535

    def test_below_minimum_is_rejected(self, monkeypatch):
        monkeypatch.setenv("PORT", "0")
        with pytest.raises(ValueError, match=r"PORT.*at least 1"):
            config.env_int("PORT", 8000, minimum=1)

    def test_above_maximum_is_rejected(self, monkeypatch):
        monkeypatch.setenv("PORT", "70000")
        with pytest.raises(ValueError, match=r"PORT.*at most 65535"):
            config.env_int("PORT", 8000, maximum=65535)


class TestEnvFloat:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("HYBRID_DENSE_WEIGHT", raising=False)
        assert config.env_float("HYBRID_DENSE_WEIGHT", 0.7) == 0.7

    def test_valid_value_is_parsed(self, monkeypatch):
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.9")
        assert config.env_float("HYBRID_DENSE_WEIGHT", 0.7) == 0.9

    def test_integer_looking_value_is_parsed(self, monkeypatch):
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1")
        assert config.env_float("HYBRID_DENSE_WEIGHT", 0.7) == 1.0

    @pytest.mark.parametrize("bad", ["heavy", "0.7.1", ""])
    def test_malformed_value_names_variable_type_value_and_default(self, monkeypatch, bad):
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", bad)
        with pytest.raises(ValueError) as excinfo:
            config.env_float("HYBRID_DENSE_WEIGHT", 0.7)
        message = str(excinfo.value)
        assert "HYBRID_DENSE_WEIGHT" in message
        assert "number" in message
        assert repr(bad) in message
        assert "0.7" in message

    @pytest.mark.parametrize("nonfinite", ["nan", "NaN", "inf", "-inf", "Infinity"])
    def test_non_finite_value_is_rejected(self, monkeypatch, nonfinite):
        # float() accepts these; the old code let them through to the reranker.
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", nonfinite)
        with pytest.raises(ValueError, match=r"HYBRID_DENSE_WEIGHT.*finite"):
            config.env_float("HYBRID_DENSE_WEIGHT", 0.7)

    def test_bounds_are_inclusive(self, monkeypatch):
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.0")
        assert config.env_float("HYBRID_DENSE_WEIGHT", 0.7, minimum=0.0, maximum=1.0) == 0.0
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1.0")
        assert config.env_float("HYBRID_DENSE_WEIGHT", 0.7, minimum=0.0, maximum=1.0) == 1.0

    def test_out_of_bounds_is_rejected(self, monkeypatch):
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1.5")
        with pytest.raises(ValueError, match=r"HYBRID_DENSE_WEIGHT.*at most 1\.0"):
            config.env_float("HYBRID_DENSE_WEIGHT", 0.7, minimum=0.0, maximum=1.0)
        monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "-0.1")
        with pytest.raises(ValueError, match=r"HYBRID_DENSE_WEIGHT.*at least 0\.0"):
            config.env_float("HYBRID_DENSE_WEIGHT", 0.7, minimum=0.0, maximum=1.0)


# Every env var, the module and constant it feeds, a valid override, its parse.
CALL_SITES = [
    ("PORT", "server", "PORT", "9001", 9001),
    ("MAX_QUERY_CHARS", "server", "MAX_QUERY_CHARS", "256", 256),
    ("MAX_TOP_K", "server", "MAX_TOP_K", "7", 7),
    ("DOCS_CONTEXT_MAX_CHARS", "server", "DOCS_CONTEXT_MAX_CHARS", "8000", 8000),
    ("HYBRID_DENSE_WEIGHT", "milvus_search", "HYBRID_DENSE_WEIGHT", "0.9", 0.9),
    ("HYBRID_SPARSE_WEIGHT", "milvus_search", "HYBRID_SPARSE_WEIGHT", "0.1", 0.1),
    ("HYBRID_RRF_K", "milvus_search", "HYBRID_RRF_K", "100", 100),
    ("AUTO_TEMPORAL_CANDIDATE_DEPTH", "intent_router", "LATEST_RELEASE_FETCH_LIMIT", "80", 80),
    ("AUTO_BM25_CANDIDATE_DEPTH", "intent_router", "KEYWORD_FETCH_LIMIT", "40", 40),
]

# Values the old code accepted at startup that break downstream; now fail fast.
OUT_OF_RANGE = [
    ("PORT", "server", "0"),  # binds a random port, not the Service's
    ("PORT", "server", "65536"),  # no such TCP port
    ("MAX_QUERY_CHARS", "server", "0"),  # rejects every query
    ("MAX_TOP_K", "server", "0"),  # Milvus limit=0 at query time
    ("DOCS_CONTEXT_MAX_CHARS", "server", "0"),  # compressor budget of 0 chars
    ("HYBRID_DENSE_WEIGHT", "milvus_search", "1.5"),  # Milvus wants [0, 1]
    ("HYBRID_SPARSE_WEIGHT", "milvus_search", "-0.1"),  # Milvus wants [0, 1]
    ("HYBRID_RRF_K", "milvus_search", "16384"),  # Milvus wants k < 16384
    ("HYBRID_RRF_K", "milvus_search", "0"),  # Milvus wants k > 0
    ("AUTO_TEMPORAL_CANDIDATE_DEPTH", "intent_router", "0"),  # silently falls back to top_k
    ("AUTO_BM25_CANDIDATE_DEPTH", "intent_router", "-5"),  # Milvus limit=-5 at query time
]


@pytest.mark.parametrize("env_name,module_name,constant,good,expected", CALL_SITES)
def test_call_site_reads_valid_value(fresh_consumers, env_name, module_name, constant, good, expected):
    fresh_consumers.setenv(env_name, good)
    module = importlib.import_module(module_name)
    assert getattr(module, constant) == expected


@pytest.mark.parametrize("env_name,module_name,constant,good,expected", CALL_SITES)
def test_call_site_rejects_malformed_value(fresh_consumers, env_name, module_name, constant, good, expected):
    fresh_consumers.setenv(env_name, "not-a-number")
    with pytest.raises(ValueError) as excinfo:
        importlib.import_module(module_name)
    assert env_name in str(excinfo.value)


@pytest.mark.parametrize("env_name,module_name,bad", OUT_OF_RANGE)
def test_call_site_rejects_out_of_range_value(fresh_consumers, env_name, module_name, bad):
    fresh_consumers.setenv(env_name, bad)
    with pytest.raises(ValueError) as excinfo:
        importlib.import_module(module_name)
    assert env_name in str(excinfo.value)


def test_defaults_match_the_previous_hardcoded_values(fresh_consumers):
    for env_name in ALL_ENV_VARS:
        fresh_consumers.delenv(env_name, raising=False)
    server = importlib.import_module("server")
    milvus_search = importlib.import_module("milvus_search")
    intent_router = importlib.import_module("intent_router")

    assert server.PORT == 8000
    assert server.MAX_QUERY_CHARS == 512
    assert server.MAX_TOP_K == 20
    assert server.DOCS_CONTEXT_MAX_CHARS == 4000
    assert milvus_search.HYBRID_DENSE_WEIGHT == 0.7
    assert milvus_search.HYBRID_SPARSE_WEIGHT == 0.3
    assert milvus_search.HYBRID_RRF_K == 60
    assert intent_router.LATEST_RELEASE_FETCH_LIMIT == 50
    assert intent_router.KEYWORD_FETCH_LIMIT == 30
    # The trailing aliases at the bottom of intent_router.py still track them.
    assert intent_router.AUTO_TEMPORAL_CANDIDATE_DEPTH == 50
    assert intent_router.AUTO_BM25_CANDIDATE_DEPTH == 30
