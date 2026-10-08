"""The suite must never reach the real network (conftest autouse guard)."""
import socket

import pytest


def test_non_loopback_connect_is_blocked():
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="must not use the network"):
            s.connect(("203.0.113.5", 443))
    finally:
        s.close()


def test_dns_lookup_is_blocked():
    with pytest.raises(RuntimeError, match="must not use the network"):
        socket.getaddrinfo("api.groq.com", 443)
