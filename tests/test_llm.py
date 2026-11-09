import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from finserv_rca.config import LLMConfig, ProviderConfig
from finserv_rca.llm.providers import GenerationRequest, OpenAICompatibleProvider, ProviderError
from finserv_rca.llm.router import AllProvidersFailed, ModelRouter


class Fake(BaseHTTPRequestHandler):
    reply = "Tier 1 must be at least 6% of RWA. [S1]"
    status = 200
    calls = 0

    def do_POST(self):
        Fake.calls += 1
        self.rfile.read(int(self.headers["Content-Length"]))
        body = {
            "choices": [{"message": {"content": Fake.reply}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        self.send_response(Fake.status)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    Fake.status, Fake.calls = 200, 0
    Fake.reply = "Tier 1 must be at least 6% of RWA. [S1]"
    httpd = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def local(url, name="local", **kwargs):
    return ProviderConfig(name=name, kind="openai_compatible", base_url=url, model="m", **kwargs)


def extractive():
    return ProviderConfig(name="extractive", kind="extractive")


REQUEST = GenerationRequest(system="s", prompt="p", question="q")


def test_openai_compatible_contract(server):
    result = OpenAICompatibleProvider(local(server)).generate(REQUEST)
    assert result.text.startswith("Tier 1") and result.input_tokens == 100


def test_provider_error_on_http_500(server):
    Fake.status = 500
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(local(server)).generate(REQUEST)


def test_fallback_marks_answer_degraded(server):
    Fake.status = 500
    router = ModelRouter(LLMConfig(generation_chain=["local", "extractive"],
                                   providers=[local(server), extractive()]))
    result = router.generate("generation", REQUEST, {"GLOBAL"})
    assert result.provider == "extractive" and result.degraded


def test_external_provider_never_gets_rbi_text(server):
    config = local(server, name="external", hosting="external", region="US",
                   allowed_residency=["GLOBAL"])
    router = ModelRouter(LLMConfig(generation_chain=["external", "extractive"],
                                   providers=[config, extractive()]))
    result = router.generate("generation", REQUEST, {"IN"})
    assert Fake.calls == 0 and result.provider == "extractive"
    assert router.denials


def test_denied_with_no_fallback_fails_closed(server):
    config = local(server, name="external", hosting="external", allowed_residency=["GLOBAL"])
    router = ModelRouter(LLMConfig(generation_chain=["external"], providers=[config]))
    with pytest.raises(AllProvidersFailed):
        router.generate("generation", REQUEST, {"EU"})
    assert Fake.calls == 0


def test_circuit_breaker_stops_calling_failed_provider(server):
    Fake.status = 500
    router = ModelRouter(LLMConfig(generation_chain=["local", "extractive"],
                                   breaker_failure_threshold=2,
                                   providers=[local(server), extractive()]))
    for _ in range(4):
        router.generate("generation", REQUEST, {"GLOBAL"})
    assert Fake.calls == 2


def test_budget_stops_external_provider(server):
    config = local(server, name="external", hosting="external", cost_per_1k_input=1000.0)
    router = ModelRouter(LLMConfig(generation_chain=["external", "extractive"], daily_budget_usd=1.0,
                                   providers=[config, extractive()]))
    router.generate("generation", REQUEST, {"GLOBAL"})
    assert router.generate("generation", REQUEST, {"GLOBAL"}).provider == "extractive"


def test_rag_retries_then_falls_back_when_draft_invents_a_number(fresh, server):
    fresh.router = ModelRouter(LLMConfig(generation_chain=["local", "extractive"],
                                         providers=[local(server), extractive()]))
    fresh.rag.router = fresh.router
    Fake.reply = "Tier 1 capital must be at least 99% of RWA. [S1]"
    answer = fresh.rag.ask("What is the minimum Tier 1 capital ratio?")
    assert Fake.calls == 2
    assert "99%" not in answer.answer and answer.degraded
    assert any("fell back" in n for n in answer.guardrails.notes)
