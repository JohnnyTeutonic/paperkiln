#!/usr/bin/env python3
"""Re-adopt or release orphaned Colab sessions.

The colab CLI keeps each session's access key in ~/.config/colab-cli/
sessions.json. When one command against a vm returns 404/401 (a flaky
tunnel, an expired hourly token) the CLI declares the session "lost" and
deletes that record. The vm keeps running, `colab sessions` lists it as
"[?] <endpoint>", and it holds one of the three concurrent GPU slots
until Colab's idle timeout reclaims it. Seen repeatedly 12-13 Sep 2026.

Colab's own assignment listing returns a FRESH key for every vm the
account holds, orphans included, so an orphan can be put back under a
name with its work intact. This script does that with the CLI's own
library, so it must run with the CLI's interpreter:

    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_adopt.py --list
    ... --adopt <endpoint>=<name>      # re-register under a name (+ keep-alive)
    ... --release <endpoint>           # unassign (stop) a vm by endpoint
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list assignments")
    ap.add_argument("--adopt", action="append", default=[],
                    metavar="ENDPOINT=NAME")
    ap.add_argument("--release", action="append", default=[],
                    metavar="ENDPOINT")
    ap.add_argument("--refresh", action="store_true",
                    help="replace every registered session's stored key "
                         "with a fresh one from the listing")
    args = ap.parse_args()

    from colab_cli.common import state
    from colab_cli.state import SessionState
    from colab_cli.commands.session import spawn_keep_alive

    sessions = state.store.list()
    known = {s.endpoint: n for n, s in sessions.items()}
    assignments = state.client.list_assignments()
    for a in assignments:
        tag = known.get(a.endpoint, "?")
        ttl = getattr(a.runtime_proxy_info, "token_expires_in_seconds", -1)
        print(f"[{tag}] {a.endpoint} {a.accelerator.value} token_ttl={ttl}s")

    if args.refresh:
        # The key the CLI stores at `colab new` is static and expires after
        # about an hour; the next request then gets 401, the CLI declares
        # the session lost and deletes the record (the orphan mechanism).
        # The listing mints a fresh key on every call, so re-storing it
        # keeps a session usable for as long as the vm lives.
        by_ep = {a.endpoint: a for a in assignments}
        for name, s in sessions.items():
            a = by_ep.get(s.endpoint)
            if a is None:
                continue
            cur = state.store.get(name)
            if cur is None:
                continue
            cur.token = a.runtime_proxy_info.token
            cur.url = a.runtime_proxy_info.url
            state.store.add(cur)
            print(f"refreshed key for {name} ({s.endpoint})")

    wanted = dict(kv.split("=", 1) for kv in args.adopt)
    for a in assignments:
        if a.endpoint not in wanted:
            continue
        name = wanted[a.endpoint]
        if a.endpoint in known:
            print(f"{a.endpoint} already registered as {known[a.endpoint]}")
            continue
        if state.store.get(name) is not None:
            print(f"name {name} already in use; not adopting {a.endpoint}")
            continue
        s = SessionState(name=name,
                         token=a.runtime_proxy_info.token,
                         url=a.runtime_proxy_info.url,
                         endpoint=a.endpoint,
                         variant=a.variant.name,
                         accelerator=a.accelerator.value)
        state.store.add(s)
        s.keep_alive_pid = spawn_keep_alive(
            a.endpoint, name,
            auth_provider=getattr(state, "auth_provider", None),
            config_path=getattr(state, "config_path", None))
        state.store.add(s)
        print(f"adopted {a.endpoint} as {name} (keep-alive pid {s.keep_alive_pid})")

    for ep in args.release:
        state.client.unassign(ep)
        print(f"released {ep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
