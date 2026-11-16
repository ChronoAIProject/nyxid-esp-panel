#!/usr/bin/env python3
"""Build one board/environment in its own sdkconfig and output directory."""
import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from board_profiles import ROOT, catalog


def command(profile, environment, api_url=None):
    build = ROOT / "build/profiles" / profile / environment
    args = ["idf.py", "-B", str(build), "-D", f"PANEL_BOARD_PROFILE={profile}",
            "-D", f"SDKCONFIG={build / 'sdkconfig'}"]
    if environment == "local":
        if not api_url:
            raise ValueError("Local builds require --api-url (a LAN NyxID origin)")
        origin = urlsplit(api_url)
        if origin.scheme != "http" or not origin.hostname or origin.username or origin.password or origin.path or origin.query or origin.fragment:
            raise ValueError("--api-url must be a plain LAN HTTP origin")
        args += ["-D", "NYX_PANEL_NODE_DEV_MODE=ON", "-D", f"NYX_PANEL_AGENT_DEV_API_URL={api_url}",
                 "-D", f"NYX_PANEL_AGENT_DEV_LOGIN_URL=http://{origin.hostname}:3000/login/agent-key"]
    else:
        if api_url:
            raise ValueError("Production builds cannot use a development API URL")
        args += ["-D", "NYX_PANEL_NODE_DEV_MODE=OFF", "-D", "NYX_PANEL_AGENT_DEV_API_URL=",
                 "-D", "NYX_PANEL_AGENT_DEV_LOGIN_URL="]
    return build, args + ["build"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=catalog()["profiles"], required=True)
    parser.add_argument("--environment", choices=("production", "local"), default="production")
    parser.add_argument("--api-url")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    try:
        build, cmd = command(args.profile, args.environment, args.api_url)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"build": str(build), "command": cmd}), flush=True)
    if not args.print_command:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
