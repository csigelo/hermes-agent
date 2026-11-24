"""Plugin A2A peer authenticators (``ctx.register_a2a_peer_authenticator``, #131484).

A provider is consulted before the static A2A tokens: a str is the peer identity, None defers, and
any raise is a 401 that never falls through to the static tokens. A provider with ``challenge`` /
``accept`` gets the two unauthenticated ``peer-auth`` routes it mints its own credential on.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import http.client
import json
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import security
from plugins.platforms.a2a.security import PeerAuthError

DID = "did:key:z6MkPeer"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("A2A_PEER_TOKENS", "A2A_BEARER_TOKEN", "A2A_HOST", "A2A_TRUSTED_PEERS",
                 "A2A_ALLOW_ALL_USERS", "A2A_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    security.peer_authenticators.reset_for_tests()
    yield
    security.peer_authenticators.reset_for_tests()


def _register(provider):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    ctx = PluginContext(PluginManifest(name="probe", key="probe", source="user"), PluginManager())
    return ctx.register_a2a_peer_authenticator(provider)


class Fixed:
    """``authenticate`` returns (or raises) a fixed outcome and records each call's agent."""

    def __init__(self, outcome, name="fixed"):
        self.name, self.outcome, self.agents = name, outcome, []

    def authenticate(self, authorization, client_ip, agent):
        self.agents.append(agent)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class Minting:
    """Challenge/accept provider that mints its own tokens (core keeps no token store)."""

    name = "minting"

    def __init__(self):
        self.tokens, self.challenge_agents = {}, []

    def challenge(self, did, agent):
        if not did.startswith("did:"):
            raise ValueError("did: want a DID")
        self.challenge_agents.append(agent.get("slug") or "default")
        return {"did": did, "nonce": "n-1", "ctx": agent.get("slug") or "default"}

    def accept(self, body, agent):
        if body.get("sig") != "good":
            raise PeerAuthError("bad signature")
        token = f"mint_{len(self.tokens)}"
        self.tokens[token] = body["did"]
        return {"identity": body["did"], "token": token, "expires_in": 3600}

    def authenticate(self, authorization, client_ip, agent):
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token.startswith("mint_"):
            if token not in self.tokens:
                raise PeerAuthError("unknown or expired token")
            return self.tokens[token]
        return None  # not ours: defer


def test_provider_defer_keeps_static_tokens(monkeypatch):
    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
    provider = Fixed(None)
    assert _register(provider) is not None
    ctx = security.A2ASecurityContext.capture()
    assert ctx.authenticate("Bearer tok-a", "10.0.0.8", {"slug": ""}) == "alice"
    assert ctx.authenticate("Bearer tok-a", "10.0.0.8") == "alice"  # legacy two-argument call
    assert ctx.authenticate("Bearer wrong", "10.0.0.8") is None
    assert provider.agents[0] == {"slug": ""} and provider.agents[1] == {}


def test_provider_identity_goes_through_the_trust_gate(monkeypatch):
    monkeypatch.setenv("A2A_TRUSTED_PEERS", DID)
    _register(Fixed(DID))
    ctx = security.A2ASecurityContext.capture()
    identity = ctx.authenticate("Bearer anything", "10.0.0.8", {"slug": ""})
    assert identity == DID and ctx.is_trusted_peer(identity)
    assert not ctx.is_trusted_peer("did:key:z6MkOther")


@pytest.mark.parametrize("error", [PeerAuthError("expired"), RuntimeError("provider bug")])
def test_provider_raise_rejects_without_trying_static_tokens(monkeypatch, error):
    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
    monkeypatch.setenv("A2A_BEARER_TOKEN", "shared")
    _register(Fixed(error))
    ctx = security.A2ASecurityContext.capture()
    assert ctx.authenticate("Bearer tok-a", "10.0.0.8", {}) is None
    assert ctx.authenticate("Bearer shared", "10.0.0.8", {}) is None


def test_non_string_identity_is_rejected():
    _register(Fixed(""))
    assert security.A2ASecurityContext.capture().authenticate("Bearer x", "10.0.0.8", {}) is None


def test_a_provider_is_a_credential_for_bind_safety(monkeypatch):
    monkeypatch.setenv("A2A_HOST", "0.0.0.0")
    bare = security.A2ASecurityContext.capture()
    assert bare.localhost_only() and bare.resolve_bind_host() == "127.0.0.1"
    _register(Fixed(None))
    ctx = security.A2ASecurityContext.capture()
    assert not ctx.localhost_only() and ctx.resolve_bind_host() == "0.0.0.0"
    assert ctx.dispatch_fails_closed()  # still needs A2A_TRUSTED_PEERS (or allow-all) on a wide bind


def test_registration_needs_authenticate_and_unload_removes_it():
    class NoAuth:
        name = "noauth"

    assert _register(NoAuth()) is None
    handle = _register(Fixed(None, name="leased"))
    assert [p.name for p in security.A2ASecurityContext.capture().peer_authenticators] == ["leased"]
    handle.dispose()
    assert security.A2ASecurityContext.capture().peer_authenticators == ()


def _status(method, url, body=None, headers=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _live(monkeypatch, provider, agents=None):
    from gateway.config import PlatformConfig
    from tests.plugins.test_a2a_plugin import _make_live_adapter

    _register(provider)
    adapter, base = _make_live_adapter(monkeypatch)
    if agents:  # served agents for per-agent peer-auth paths (re-load with the routing config)
        adapter._agents = adapter._load_served_agents({"agents": agents})
    return adapter, base


class JwtLike:
    """Authenticate-only provider (a Keycard-style JWT): no challenge/accept, HS256 for the test."""

    name = "jwt"
    key = b"issuer-key"

    @classmethod
    def mint(cls, sub, key=None):
        enc = lambda obj: base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
        head = f"{enc({'alg': 'HS256'})}.{enc({'sub': sub})}"
        sig = hmac.new(key or cls.key, head.encode(), hashlib.sha256).digest()
        return f"{head}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"

    def authenticate(self, authorization, client_ip, agent):
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not token.startswith("eyJ"):
            return None  # not a JWT: defer
        head, _, sig = token.rpartition(".")
        good = base64.urlsafe_b64encode(hmac.new(self.key, head.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(sig, good):
            raise PeerAuthError("bad JWT signature")
        payload = head.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["sub"]


def test_an_authenticate_only_provider_coexists_with_a_minting_one(monkeypatch):
    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
    jwt, minting = JwtLike(), Minting()
    _register(jwt)
    _register(minting)
    ctx = security.A2ASecurityContext.capture()
    assert [p.name for p in ctx.peer_authenticators] == ["jwt", "minting"]  # registration order
    minting.tokens["mint_0"] = DID
    assert ctx.authenticate("Bearer " + JwtLike.mint("spiffe://acme/agent"), "10.0.0.8", {}) == "spiffe://acme/agent"
    assert ctx.authenticate("Bearer mint_0", "10.0.0.8", {}) == DID  # jwt deferred, minting answered
    assert ctx.authenticate("Bearer tok-a", "10.0.0.8", {}) == "alice"  # both deferred
    assert ctx.authenticate("Bearer " + JwtLike.mint("alice", key=b"forged"), "10.0.0.8", {}) is None
    assert ctx.authenticate("Bearer mint_7", "10.0.0.8", {}) is None


def test_challenge_accept_round_trip_over_http(monkeypatch):
    from tests.plugins.test_a2a_plugin import _send_body

    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
    monkeypatch.setenv("A2A_TRUSTED_PEERS", f"{DID},alice")
    provider = Minting()
    adapter, base = _live(monkeypatch, provider, agents={"helper": {"profile": "default"}})

    def exercise():
        assert _status("GET", base + "/peer-auth/challenge")[0] == 400  # no did
        assert _status("GET", base + "/peer-auth/challenge?did=x") == (400, {"error": "did: want a DID"})  # ValueError
        conn = http.client.HTTPConnection(base.split("//")[1], timeout=15)
        conn.putrequest("POST", "/peer-auth")
        conn.putheader("Content-Length", "-1")  # would read to EOF
        conn.endheaders()
        assert conn.getresponse().status == 400
        conn.close()
        code, challenge = _status("GET", base + f"/helper/peer-auth/challenge?did={DID}")
        assert code == 200 and challenge == {"did": DID, "nonce": "n-1", "ctx": "helper"}
        assert _status("POST", base + "/peer-auth", {"did": DID, "sig": "bad"})[0] == 401
        assert _status("POST", base + "/peer-auth", b"[1]")[0] == 400
        assert _status("POST", base + "/peer-auth", b"{not json")[0] == 400
        code, minted = _status("POST", base + "/peer-auth", {"did": DID, "sig": "good"})
        assert code == 200 and minted == {"identity": DID, "token": "mint_0", "expires_in": 3600}

        auth = {"Authorization": "Bearer " + minted["token"]}
        code, reply = _status("POST", base + "/", _send_body("hi"), auth)
        assert code == 200 and reply["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert _status("POST", base + "/", _send_body("hi"), {"Authorization": "Bearer mint_9"})[0] == 401
        assert _status("POST", base + "/", _send_body("hi"), {"Authorization": "Bearer tok-a"})[0] == 200
        assert _status("POST", base + "/", _send_body("hi"))[0] == 401

        # The authenticated GET lists served agents for the provider identity only.
        assert "served_agents" in _status("GET", base + "/", headers=auth)[1]
        assert "served_agents" not in _status("GET", base + "/", headers={"Authorization": "Bearer mint_9"})[1]

    async def run():
        assert await adapter.connect() is True
        try:
            await asyncio.to_thread(exercise)
        finally:
            await adapter.disconnect()

    asyncio.run(run())
    assert provider.challenge_agents == ["helper"]  # the routed agent reaches the provider


def test_rate_limiter_drops_idle_buckets(monkeypatch):
    from types import SimpleNamespace

    from plugins.platforms.a2a import protocol

    clock = [1000.0]
    monkeypatch.setattr(protocol, "time", SimpleNamespace(time=lambda: clock[0]))
    limiter = protocol.RateLimiter()
    for i in range(100):
        assert limiter.allow(f"ip:10.0.0.{i}")
    clock[0] += 61
    assert limiter.allow("ip:10.0.1.1") and list(limiter._buckets) == ["ip:10.0.1.1"]


def test_peer_auth_routes_are_rate_limited_per_client_ip(monkeypatch):
    monkeypatch.setenv("A2A_RATE_LIMIT", "1")
    adapter, base = _live(monkeypatch, Minting())

    def exercise():
        assert _status("GET", base + f"/peer-auth/challenge?did={DID}")[0] == 200
        assert _status("GET", base + f"/peer-auth/challenge?did={DID}")[0] == 429

    async def run():
        assert await adapter.connect() is True
        try:
            await asyncio.to_thread(exercise)
        finally:
            await adapter.disconnect()

    asyncio.run(run())


def test_provider_without_challenge_accept_serves_no_peer_auth_routes(monkeypatch):
    adapter, base = _live(monkeypatch, Fixed(None))

    def exercise():
        assert _status("GET", base + f"/peer-auth/challenge?did={DID}")[0] == 404
        assert _status("POST", base + "/peer-auth", {"did": DID})[0] == 404

    async def run():
        assert await adapter.connect() is True
        try:
            await asyncio.to_thread(exercise)
        finally:
            await adapter.disconnect()

    asyncio.run(run())


def test_tenant_routing_to_another_agent_is_judged_by_the_provider_again(monkeypatch):
    """A credential the provider scoped to the root agent must not reach a served agent via ``params.tenant``."""
    from tests.plugins.test_a2a_plugin import _send_body

    monkeypatch.setenv("A2A_TRUSTED_PEERS", DID)

    class RootOnly(Fixed):
        def authenticate(self, authorization, client_ip, agent):
            super().authenticate(authorization, client_ip, agent)
            return DID if agent.get("slug") == "" else None

    adapter, base = _live(monkeypatch, RootOnly(None), agents={"helper": {"profile": "default"}})
    auth = {"Authorization": "Bearer root-scoped"}

    def exercise():
        assert _status("POST", base + "/", _send_body("hi"), auth)[0] == 200
        assert _status("POST", base + "/", _send_body("hi", extra_params={"tenant": "helper"}), auth)[0] == 401

    async def run():
        assert await adapter.connect() is True
        try:
            await asyncio.to_thread(exercise)
        finally:
            await adapter.disconnect()

    asyncio.run(run())
