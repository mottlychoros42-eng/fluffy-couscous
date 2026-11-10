"""
ntfy push notifications for AirBnBrrr.

One alert event = ONE short notification: the worst problems in plain words, then what to do.
Since v3.0 an alert comes from ONE source (ML model, Safety limits or Sensor offline) and the title says which.
The full details (all readings, every step) are on the Alerts page.

Configure with environment variables on Render:
    NTFY_TOPIC    required. Use a long random name, public topics are readable by anyone who
                  knows the name, e.g. AirBnrr_alien_test_code_k8x2m9qv4t
    NTFY_SERVER   optional, default https://ntfy.sh (set it if you self-host)
    NTFY_TOKEN    optional access token, for protected topics
    APP_URL       optional, your site address; tapping the notification opens the Alerts page
Uses only the standard library, so there is nothing new to install.
"""
import http.client
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from services import action_guide

SHOW_PROBLEMS = 3     # problems listed in the notification
MAX_BYTES = 1500


def finding_text(f):
    """One plain line for a rule-engine finding (dict) or a plain string."""
    if isinstance(f, str):
        return f.strip()
    if isinstance(f, dict):
        return action_guide.plain_line(f)
    return str(f)


_SEV = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def build_problems(critical_reasons, findings, ml):
    """Plain problem lines, worst first, no duplicates (the AI note goes last).

    The rule engine uses the same limits as the instant safety alerts, so when it has findings the raw
    limit lines would only repeat them; they are used only when the rule engine found nothing.
    """
    ordered = sorted(findings or [], key=lambda f: -_SEV.get(f.get("severity") if isinstance(f, dict) else None, 0))
    items = [finding_text(f) for f in ordered] if ordered else list(critical_reasons or [])
    line = action_guide.ml_line(ml)
    if line:
        items.append(line)
    seen, out = set(), []
    for it in items:
        key = it.lower()
        if it and key not in seen:
            seen.add(key)
            out.append(it)
    return out


SOURCE_TITLE = {"ml": "ML", "safety": "Safety", "sensor": "Sensor"}
SOURCE_TAGS = {"ml": "robot", "safety": "rotating_light", "sensor": "electric_plug"}


def build_message(unit_title, is_critical, problems, readings=None, actions=None, source=None, assessment=None):
    """Return (title, body). Title is plain ASCII because it travels in an HTTP header.

    The title says where the alert comes from: 'Check (ML): AC1', 'STOP (Safety): AC1', 'Check (Sensor): AC1'.
    An ML alert (`assessment` given) lists the odd sensors with their real values and the AC's usual range, the
    severity, the main risk and what to do, like a rule-based alert does.
    """
    tag = SOURCE_TITLE.get(source)
    head = ("STOP" if is_critical else "Check") + (f" ({tag})" if tag else "")
    title = f"{head}: {unit_title}".encode("ascii", "replace").decode("ascii")
    if assessment:
        plain = ("+", "Likely cause", "Risk")                       # these are not sensors: no bullet
        lines = [assessment["lines"][0]] + [(l if l.startswith(plain) else f"- {l}") for l in assessment["lines"][1:]]
        lines.append("Why: " + (assessment.get("why_short") or "the model flagged this reading as abnormal for this AC").rstrip(".") + ".")
    else:
        ai = [p for p in problems if p.startswith("ML model")]
        real = [p for p in problems if not p.startswith("ML model")]
        lines = [f"- {p}" for p in real[:SHOW_PROBLEMS]]
        if len(real) > SHOW_PROBLEMS:
            lines.append(f"+ {len(real) - SHOW_PROBLEMS} more on the Alerts page")
        if ai:
            lines.append(ai[0])
        if not lines:
            lines.append("The AC is acting differently from its normal.")
    if actions:
        lines += ["", "Do this now:"] + [f"{i}. {a}" for i, a in enumerate(actions, start=1)]
    body = "\n".join(lines)
    return title, body.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")


RETRY_WAITS = (0, 1.5, 4.0)       # seconds before each attempt: network errors, HTTP 5xx and 429 are retried
# If the server could not reach ntfy AT ALL (no network route, DNS down, timeouts) the alert is not thrown away: it is kept and
# tried again in the background after these many seconds, for about 7 minutes. A phone notification a minute late beats none.
RETRY_QUEUE_DELAYS = (15, 30, 60, 120, 240)
WORKER_TICK = 2.0
MAX_PENDING = 20
_status_lock = threading.Lock()
_status = {"configured": None, "last_ok": None, "last_error": None, "last_attempt_at": None, "last_http_status": None,
           "last_title": None, "sent": 0, "failed": 0, "network": None, "warning": None}
_pending = []
_pending_lock = threading.Lock()
_worker = {"thread": None}

# replaceable in tests
_getaddrinfo = socket.getaddrinfo
_make_socket = socket.socket


def _record(ok, title, http_status=None, error=None):
    with _status_lock:
        _status["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
        _status["last_title"] = title
        _status["last_http_status"] = http_status
        _status["last_ok"] = ok
        _status["last_error"] = None if ok else error
        _status["sent" if ok else "failed"] += 1
        if ok:
            _status["network"] = None


def status():
    """What the dashboard shows: is ntfy set up, and did the last notification arrive (and if not, why)."""
    topic = _topic()
    with _status_lock:
        out = dict(_status)
    with _pending_lock:
        out["queued"] = len(_pending)
    out["configured"] = bool(topic)
    out["token_set"] = bool(_auth_header())
    out["topic_hint"] = (topic[:4] + "…") if topic else None
    out["server"] = _server()
    return out


def _topic():
    """The topic from NTFY_TOPIC, tolerating spaces and quotes pasted by mistake."""
    return os.environ.get("NTFY_TOPIC", "").strip().strip("\"'").strip().strip("/")


def _server():
    """NTFY_SERVER with https:// added when it was left out (a bare 'ntfy.sh' is not a usable address)."""
    server = os.environ.get("NTFY_SERVER", "").strip().strip("\"'").rstrip("/") or "https://ntfy.sh"
    if "://" not in server:
        server = "https://" + server
    return server


# ------------------------------------------------------------------------------------------- network
def _family(fam):
    return "IPv4" if fam == socket.AF_INET else "IPv6"


def _connect(host, port, timeout, use_tls):
    """Open a connection: IPv4 addresses first, then IPv6, each with its own short timeout. If nothing works the error lists
    EVERY address that was tried and why, so the log shows whether the whole network is down or just one address family."""
    try:
        infos = _getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    except OSError as e:
        raise OSError(f"cannot look up {host} (DNS): {e}")
    infos = sorted(infos, key=lambda a: 0 if a[0] == socket.AF_INET else 1)
    errors = []
    for fam, typ, proto, _canon, sa in infos:
        sock = None
        try:
            sock = _make_socket(fam, typ, proto)
            sock.settimeout(min(timeout, 5))
            sock.connect(sa)
            sock.settimeout(timeout)
            if use_tls:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            return sock
        except OSError as e:
            errors.append(f"{sa[0]} ({_family(fam)}): {e}")
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    raise OSError(f"could not connect to {host}:{port}: " + "; ".join(errors))


def _http_post(url, headers, data, timeout=8):
    """POST and return (status, body text). Raises OSError when the server cannot be reached at all."""
    u = urllib.parse.urlsplit(url)
    tls = u.scheme == "https"
    port = u.port or (443 if tls else 80)
    sock = _connect(u.hostname, port, timeout, tls)
    conn = http.client.HTTPConnection(u.hostname, port, timeout=timeout)
    conn.sock = sock                                   # already connected (and wrapped in TLS): http.client just uses it
    try:
        conn.request("POST", u.path or "/", body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read(400).decode("utf-8", "replace").strip()
    finally:
        conn.close()


def _explain_network_error(e):
    text = f"{type(e).__name__}: {e}"
    if "Errno 101" in text or "Network is unreachable" in text:
        return ("No network route from this server to ntfy (Errno 101: Network is unreachable). This is not a topic or setting "
                "problem: the server itself had no outbound connection at that moment. " + text)
    if "cannot look up" in text or "Name or service not known" in text or "Temporary failure in name resolution" in text:
        return "This server could not look up the ntfy address (DNS), so its network was down or blocked. " + text
    if "timed out" in text:
        return "The connection to ntfy timed out: the server's network is very slow or something blocks it. " + text
    return text


def diagnose_network(timeout=4):
    """Say WHY ntfy cannot be reached: DNS, then a connection test to every address. Returns {ok, verdict, lines}."""
    server = _server()
    u = urllib.parse.urlsplit(server)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    lines = []
    try:
        infos = sorted(_getaddrinfo(host, port, 0, socket.SOCK_STREAM), key=lambda a: 0 if a[0] == socket.AF_INET else 1)
    except OSError as e:
        out = {"ok": False, "host": host, "lines": [f"DNS lookup of {host}: FAILED ({e})"],
               "verdict": f"This server cannot look up {host}. Its network is down or DNS is blocked, so no notification can leave it."}
        with _status_lock:
            _status["network"] = out
        return out
    seen, ok_v4, ok_v6, tried_v4, tried_v6 = set(), False, False, 0, 0
    for fam, typ, proto, _c, sa in infos:
        if sa[0] in seen:
            continue
        seen.add(sa[0])
        sock = None
        try:
            sock = _make_socket(fam, typ, proto)
            sock.settimeout(timeout)
            sock.connect(sa)
            lines.append(f"{_family(fam)} {sa[0]}: connected")
            ok_v4, ok_v6 = ok_v4 or fam == socket.AF_INET, ok_v6 or fam != socket.AF_INET
        except OSError as e:
            lines.append(f"{_family(fam)} {sa[0]}: FAILED ({e})")
        finally:
            tried_v4 += fam == socket.AF_INET
            tried_v6 += fam != socket.AF_INET
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    if ok_v4 or ok_v6:
        verdict = (f"{host} is reachable from this server" + ("" if ok_v4 else " (IPv6 only)") +
                   ". If a notification still fails, the problem is not the network: see the error above.")
        if tried_v6 and not ok_v6 and ok_v4:
            verdict += " IPv6 has no route here, which is harmless: IPv4 is tried first."
    else:
        verdict = (f"{host} is NOT reachable from this server on any address (DNS works, every connection failed). "
                   "The hosting service has no working outbound connection to it right now. Check the host's status page or restart the service; "
                   "queued alerts are retried automatically.")
    out = {"ok": bool(ok_v4 or ok_v6), "host": host, "lines": lines, "verdict": verdict}
    with _status_lock:
        _status["network"] = out
    return out


# ------------------------------------------------------------------------------------------- sending
def _auth_header():
    """Authorization header from NTFY_TOKEN, tolerating what people paste: quotes, spaces, a leading 'Bearer ', or user:password."""
    raw = os.environ.get("NTFY_TOKEN", "").strip().strip("\"'").strip()
    if not raw:
        return None
    if raw.lower().startswith(("bearer ", "basic ")):
        return raw
    if ":" in raw and not raw.startswith("tk_"):          # username:password -> HTTP Basic (ntfy supports it)
        import base64
        return "Basic " + base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return "Bearer " + raw


def _build_headers(title, priority, tags):
    base = {"Title": title.replace("\n", " ")[:200], "Priority": str(priority),
            "User-Agent": "AirBnBrrr/3.5 (+ntfy)", "Content-Type": "text/plain; charset=utf-8"}
    auth = _auth_header()
    if auth:
        base["Authorization"] = auth
    optional = {"Tags": tags}                         # nice-to-have headers: dropped if the server rejects the request (HTTP 400)
    app_url = os.environ.get("APP_URL", "").strip().strip("\"'").rstrip("/")
    if app_url:
        optional["Click"] = (app_url if "://" in app_url else "https://" + app_url) + "/pages/alerts"
    return base, {**base, **optional}


def _attempt(title, body, priority, tags):
    """Try to deliver once (with the short in-line retries). Returns (ok, http_status, error, network_only)."""
    url = f"{_server()}/{urllib.parse.quote(_topic(), safe='')}"
    base_headers, headers = _build_headers(title, priority, tags)
    data = body.encode("utf-8")
    minimal, token_dropped, last_error, http_status, only_network = False, False, "unknown error", None, True
    for attempt, wait in enumerate(RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            status_code, detail = _http_post(url, headers, data)
            http_status, only_network = status_code, False
            if 200 <= status_code < 300:
                print(f"[NTFY] sent '{title}' (HTTP {status_code}, attempt {attempt})", flush=True)
                with _status_lock:
                    _status["warning"] = ("The NTFY_TOKEN on the server was rejected, so notifications are sent WITHOUT it. They arrive because the topic "
                                          "is public. Fix or remove NTFY_TOKEN (it should start with tk_ and come from the same ntfy server).") if token_dropped else None
                return True, status_code, None, False
            has_token = "Authorization" in headers
            if status_code in (401, 403) and has_token and not token_dropped:
                # A wrong token makes ntfy reject even PUBLIC topics. An alert matters more than the token: resend without it.
                # (If the topic really is protected this fails too, and the message below says so.)
                token_dropped, headers = True, {k: v for k, v in headers.items() if k != "Authorization"}
                print(f"[NTFY] HTTP {status_code}: the NTFY_TOKEN was rejected ({detail or 'no detail'}), resending without it", flush=True)
                continue
            if status_code in (401, 403):
                tip = ("the NTFY_TOKEN is rejected and the topic needs a login: check the token starts with tk_, was created on the SAME ntfy server "
                       "and may write to this topic (or remove NTFY_TOKEN if the topic is public)") if token_dropped or has_token else \
                      "the topic needs a login: set NTFY_TOKEN to an access token (tk_...) that may write to it"
                hint = tip
            else:
                hint = {404: "wrong NTFY_SERVER address", 429: "rate limited by the ntfy server (too many messages from this IP: use NTFY_TOKEN or self-host)"
                        }.get(status_code, "")
            last_error = f"HTTP {status_code}" + (f" ({hint})" if hint else "") + (f": {detail[:200]}" if detail else "")
            if status_code == 400 and not minimal:   # the server did not like something optional (Click / Tags): resend plain
                minimal, headers = True, dict(base_headers)
                print(f"[NTFY] HTTP 400 ({detail or 'no detail'}): resending without the optional Click/Tags headers", flush=True)
                continue
            if status_code < 500 and status_code != 429:   # retrying other 4xx would not help
                break
        except OSError as e:                         # no answer at all: DNS, no route, timeout, blocked connection
            only_network = only_network and True
            last_error = _explain_network_error(e)
        print(f"[NTFY] attempt {attempt}/{len(RETRY_WAITS)} failed for '{title}': {last_error}", flush=True)
    return False, http_status, last_error, only_network


def _enqueue(title, body, priority, tags):
    with _pending_lock:
        if len(_pending) >= MAX_PENDING:
            _pending.pop(0)
        _pending.append({"title": title, "body": body, "priority": priority, "tags": tags, "n": 0,
                         "due": time.time() + RETRY_QUEUE_DELAYS[0], "born": time.time()})
        n = len(_pending)
        if _worker["thread"] is None or not _worker["thread"].is_alive():
            _worker["thread"] = threading.Thread(target=_worker_loop, name="ntfy-retry", daemon=True)
            _worker["thread"].start()
    print(f"[NTFY] could not reach ntfy: '{title}' is queued and will be retried automatically ({n} waiting)", flush=True)


def _worker_loop():
    while True:
        time.sleep(WORKER_TICK)
        with _pending_lock:
            if not _pending:
                _worker["thread"] = None
                return
            due = [m for m in _pending if m["due"] <= time.time()]
        for m in due:
            ok, st, err, net = _attempt(m["title"], m["body"], m["priority"], m["tags"])
            with _pending_lock:
                if ok:
                    _pending.remove(m)
                else:
                    m["n"] += 1
                    if m["n"] >= len(RETRY_QUEUE_DELAYS) or not net:
                        _pending.remove(m)
                    else:
                        m["due"] = time.time() + RETRY_QUEUE_DELAYS[m["n"]]
            if ok:
                print(f"[NTFY] delivered on retry {m['n'] + 1}: '{m['title']}' (it had been waiting {int(time.time() - m['born'])} s)", flush=True)
                _record(True, m["title"], st)
            elif m["n"] >= len(RETRY_QUEUE_DELAYS) or not net:
                print(f"[NTFY] GAVE UP on '{m['title']}' after {m['n']} background retries: {err}", flush=True)
                _record(False, m["title"], st, err)


def _post(title, body, priority, tags, queue=True):
    topic = _topic()
    if not topic:
        msg = "NTFY_TOPIC is not set on the server, so no notification can be sent"
        print(f"[NTFY] {msg}", flush=True)
        _record(False, title, None, msg)
        return False
    ok, http_status, last_error, only_network = _attempt(title, body, priority, tags)
    if ok:
        _record(True, title, http_status)
        return True
    print(f"[NTFY] NOT delivered '{title}': {last_error}", flush=True)
    _record(False, title, http_status, last_error)
    if queue and only_network:
        _enqueue(title, body, priority, tags)         # nothing reached the server: keep it and try again later
    return False


def send_alert(unit_title, is_critical, problems, readings=None, actions=None, source=None, assessment=None):
    title, body = build_message(unit_title, is_critical, problems, readings, actions, source, assessment)
    base = SOURCE_TAGS.get(source, "rotating_light" if is_critical else "warning")
    tags = base if source is None else f"{base},{'rotating_light' if is_critical else 'warning'}"
    return _post(title, body, 5 if is_critical else 4, tags)


def send_test():
    """A test sends once and reports honestly: it is not queued, so a failure is shown, not hidden."""
    return _post("AirBnBrrr test", "ntfy is connected. Real alerts look like this: the worst problems in plain words, then what to do.",
                 3, "white_check_mark", queue=False)


def _boot_log():
    t = _topic()
    if t:
        print(f"[NTFY] ready: topic '{t[:4]}…' ({len(t)} characters), server {_server()}", flush=True)
    else:
        print("[NTFY] NTFY_TOPIC is NOT set on this server: no phone notifications will be sent", flush=True)


_boot_log()
