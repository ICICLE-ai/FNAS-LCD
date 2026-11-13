"""Register FNAS-LCD's Expanse apps in Tapis and share them with every ICICLE user.

Run it as the account that owns the apps. It asks for that account's Tapis
password, which is never printed or saved:

    .venv/bin/python tapis/register_apps.py

For each tapis/fnas-lcd-expanse-*.app.json, it creates that app version, or
updates it to match the file if it already exists, and then makes the app
public. It first checks that the apps' image tag is the service-version in
icicle-service.yaml: the jobs run that image, so they run the same code as
the website.
"""

import getpass
import json
import re
import sys
from pathlib import Path

import requests

BASE_URL = "https://icicleai.tapis.io"
TAPIS_DIR = Path(__file__).resolve().parent
REPO = TAPIS_DIR.parent


def main() -> None:
    apps = [json.loads(p.read_text()) for p in sorted(TAPIS_DIR.glob("fnas-lcd-expanse-*.app.json"))]
    if not apps:
        sys.exit("No tapis/fnas-lcd-expanse-*.app.json files found.")
    match = re.search(r"^service-version:\s*(\S+)", (REPO / "icicle-service.yaml").read_text(), re.M)
    if not match:
        sys.exit("icicle-service.yaml has no service-version.")
    version = match.group(1)
    for app in apps:
        if not app["containerImage"].endswith(f":{version}"):
            sys.exit(f"{app['id']} runs {app['containerImage']}, but icicle-service.yaml's "
                     f"service-version is {version}. Make them match.")

    print("Apps to register, shared with every ICICLE user:")
    for app in apps:
        print(f"  {app['id']} {app['version']}  ({app['containerImage']})")
    user = input("Tapis username [sinamps]: ").strip() or "sinamps"
    password = getpass.getpass("Tapis password (not shown): ")
    resp = requests.post(f"{BASE_URL}/v3/oauth2/tokens", timeout=60,
                         json={"username": user, "password": password, "grant_type": "password"})
    del password
    if not resp.ok:
        sys.exit(f"Login failed: HTTP {resp.status_code} {resp.text[:300]}")
    headers = {"X-Tapis-Token": resp.json()["result"]["access_token"]["access_token"]}

    def call(method: str, path: str, **kwargs):
        r = requests.request(method, BASE_URL + path, headers=headers, timeout=120, **kwargs)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.ok:
            return True, body.get("result")
        return False, f"HTTP {r.status_code}: {body.get('message') or r.text[:300]}"

    for app in apps:
        name = f"{app['id']} {app['version']}"
        ok, current = call("GET", f"/v3/apps/{app['id']}/{app['version']}")
        if ok and current.get("owner") != user:
            sys.exit(f"{name} belongs to {current.get('owner')}, not {user}. Stopping.")
        if not ok:
            ok, res = call("POST", "/v3/apps", json=app)
            if not ok and "exist" in str(res).lower():
                # A deleted app keeps its name; bring it back and update it.
                call("POST", f"/v3/apps/{app['id']}/undelete")
                ok, res = call("PUT", f"/v3/apps/{app['id']}/{app['version']}", json=app)
            print(f"{name}: {'created' if ok else 'NOT created: ' + str(res)}")
        else:
            ok, res = call("PUT", f"/v3/apps/{app['id']}/{app['version']}", json=app)
            print(f"{name}: {'updated to match the file' if ok else 'NOT updated: ' + str(res)}")
        if not ok:
            sys.exit("Stopping.")
        ok, res = call("POST", f"/v3/apps/share_public/{app['id']}")
        if not ok:
            sys.exit(f"{name}: couldn't share it with every ICICLE user: {res}")
        ok, check = call("GET", f"/v3/apps/{app['id']}/{app['version']}")
        public = ok and check.get("isPublic")
        print(f"  shared with every ICICLE user: {'yes' if public else 'NO, check: ' + str(check)}")
    print("Done.")


if __name__ == "__main__":
    main()
