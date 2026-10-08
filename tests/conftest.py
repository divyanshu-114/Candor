"""Test-suite safety rails.

1. Usage ledger: fake-provider calls never reach outputs/usage_ledger.jsonl
   (env set at import AND per test).
2. No network, ever: the real OpenAI client class is replaced by one that
   raises, so a test that forgets to inject a fake client fails loudly
   instead of spending quota.
3. LLM_FROZEN_ROLES is cleared per test (tests drive fake clients); tests of
   the freeze itself set it explicitly.
"""
import ipaddress
import os
import socket
import tempfile

import pytest

os.environ["USAGE_LEDGER_PATH"] = os.path.join(tempfile.mkdtemp(prefix="candor-test-ledger-"), "usage_ledger.jsonl")


def _no_network(*_a, **_k):
    raise RuntimeError("tests must not create a real OpenAI client; inject a fake client instead")


@pytest.fixture(autouse=True)
def _safety_rails(tmp_path, monkeypatch):
    import memory.llm as llm_mod
    monkeypatch.setenv("USAGE_LEDGER_PATH", str(tmp_path / "usage_ledger.jsonl"))
    monkeypatch.delenv("LLM_FROZEN_ROLES", raising=False)
    monkeypatch.setattr(llm_mod, "OpenAI", _no_network)


# --- No real network, ever ---------------------------------------------------
_REAL_CONNECT = socket.socket.connect
_REAL_GETADDRINFO = socket.getaddrinfo


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if isinstance(address, tuple) and address and not _is_local(address[0]):
        raise RuntimeError(f"tests must not use the network: blocked connect to {address[0]!r}. "
                           "Inject a fake client or a localhost fake server instead.")
    return _REAL_CONNECT(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        raise RuntimeError(f"tests must not use the network: blocked DNS lookup of {host!r}.")
    return _REAL_GETADDRINFO(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def _block_real_sockets(monkeypatch):
    """Non-loopback sockets raise with a clear message. Loopback stays open so
    a fake HTTP server on 127.0.0.1 can stand in for a provider."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)


# --- Shared, session-scoped resources ---------------------------------------
@pytest.fixture(scope="session")
def data_store():
    """The real corpus, loaded once per session (read-only: never mutate it)."""
    from memory.store import MemoryStore
    return MemoryStore("./data")


# --- A tiny synthetic corpus shared by the retrieval / degradation tests ------------------------------------------
import json  # noqa: E402

T = "2026-09-0{d}T{h}:00:00-07:00"


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path
    for sub in ("connectors/slack", "connectors/gmail", "connectors/google_calendar", "native/dictation"):
        (d / sub).mkdir(parents=True)
    users = [{"id": "U1", "real_name": "Alex Rivera", "email": "alex@acme.example.com"},
             {"id": "U2", "real_name": "Sarah Kim", "email": "sarah.kim@acme.example.com"}]
    (d / "connectors/slack/users.json").write_text(json.dumps(users))
    (d / "connectors/slack/channels.json").write_text(json.dumps([{"id": "C1", "name": "eng", "is_dm": False, "members": ["U1", "U2"]}]))
    slack = [
        {"id": "SL-1", "channel_id": "C1", "user": "U2", "ts": T.format(d=1, h="09"), "text": "Geocoder fix is verified on staging."},
        {"id": "SL-2", "channel_id": "C1", "user": "U1", "ts": T.format(d=1, h="10"), "text": "Launch is Sep 30, I'll send the pricing plan by Friday."},
        {"id": "SL-3", "channel_id": "C1", "user": "U1", "ts": T.format(d=2, h="10"), "text": "Correction: the launch moved to Oct 14, sorry."},
        {"id": "SL-4", "channel_id": "C1", "user": "U1", "ts": T.format(d=3, h="10"), "text": "Pricing plan sent, attached."},
        {"id": "SL-5", "channel_id": "C1", "user": "U2", "ts": T.format(d=3, h="11"), "text": "Unrelated lunch chatter about tacos."},
    ]
    (d / "connectors/slack/messages.jsonl").write_text("\n".join(json.dumps(x) for x in slack))
    mails = [{"id": "EM-1", "thread_id": "TH-1", "date": T.format(d=1, h="12"), "from": "Sarah Patel <sarah.patel@acmefreight.example.com>",
              "to": ["alex@acme.example.com"], "cc": [], "subject": "Pricing proposal", "body": "Please send the pricing proposal soon.", "labels": [], "attachments": []},
             {"id": "EM-2", "thread_id": "TH-1", "date": T.format(d=2, h="12"), "from": "Alex Rivera <alex@acme.example.com>",
              "to": ["sarah.patel@acmefreight.example.com"], "cc": [], "subject": "Re: Pricing proposal", "body": "Slipping to Tuesday, sorry.", "labels": [], "attachments": []}]
    (d / "connectors/gmail/messages.jsonl").write_text("\n".join(json.dumps(x) for x in mails))
    (d / "connectors/google_calendar/events.jsonl").write_text(json.dumps(
        {"id": "CAL-1", "summary": "Board meeting", "description": "", "location": "HQ", "start": {"dateTime": "2026-09-05T09:00:00-07:00"},
         "end": {"dateTime": "2026-09-05T12:00:00-07:00"}, "organizer": "alex@acme.example.com", "attendees": [{"email": "alex@acme.example.com"}],
         "status": "confirmed", "recurrence": None, "created": T.format(d=1, h="08"), "updated": T.format(d=1, h="08")}))
    (d / "native/dictation/dictations.jsonl").write_text("")
    return d


@pytest.fixture()
def corpus(corpus_dir):
    from memory.store import MemoryStore
    return MemoryStore(str(corpus_dir))


