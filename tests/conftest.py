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
