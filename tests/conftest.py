import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("ANTHROPIC_API_KEY", None)  # tests never call the API

from stockintel.agents.base import AgentContext  # noqa: E402
from stockintel.config import Settings  # noqa: E402
from stockintel.data.synthetic import SyntheticProvider  # noqa: E402
from stockintel.llm.client import LLMClient  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return Settings(anthropic_api_key=None, backtest_folds=3, enable_lstm=False, enable_prophet=False, db_path=":memory:")


@pytest.fixture(scope="session")
def provider():
    return SyntheticProvider(end="2026-06-30")


@pytest.fixture
def ctx(provider, settings):
    sym = "AAPL"
    return AgentContext(ticker=sym, provider=provider, llm=LLMClient(settings, enabled=False), settings=settings,
                        shared={"prices": provider.history(sym), "quote": provider.quote(sym),
                                "profile": provider.profile(sym), "benchmark": provider.history("^GSPC")["Close"]})
