"""Inherit the external-screen board's local service profile and verify the linked marker app."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

import check_builds

PROFILE = "p4-rev1-16mb"
API = "NYX_PANEL_AGENT_DEV_API_URL"
LOGIN = "NYX_PANEL_AGENT_DEV_LOGIN_URL"
NODE = "NYX_PANEL_NODE_DEV_MODE"
MAC_RECORD = Path(__file__).resolve().parents[2] / "results/mac-c161-rev1-dev-marker-build.json"
PATCH = Path(__file__).resolve().with_name("c161-fixed-marker.patch")
PLAYBACK_ONLY = PATCH.with_name("fixed-playback-only.patch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cache_values(path):
    return dict(re.findall(r"^(\w+):(?:STRING|BOOL|UNINITIALIZED)=(.*)$", path.read_text(), re.M))


@dataclass(frozen=True)
class Services:
    api: str
    login: str
    node_dev: bool

    @property
    def fingerprint(self):
        return sha256(f"{self.api}|{self.login}".encode()).hexdigest()[:10]

    def cmake(self):
        return {API: self.api, LOGIN: self.login, NODE: "ON" if self.node_dev else "OFF"}


def from_cache(path):
    data = cache_values(path)
    require(data.get("PANEL_BOARD_PROFILE", PROFILE) == PROFILE, "Service cache belongs to another board profile")
    api, login = data[API], data[LOGIN]
    match = re.fullmatch(r"http://((?:10\.\d+|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+)(?::\d+)?", api)
    require(match and re.fullmatch(r"http://" + re.escape(match[1]) + r"(?::\d+)?/login/agent-key", login),
            "Expected a matching pair of private development URLs in the source cache")
    node = data[NODE].upper()
    require(node in ("1", "ON", "YES", "TRUE", "Y", "0", "OFF", "NO", "FALSE", "N", ""),
            "Unrecognized node mode in the source cache")
    return Services(api, login, node in ("1", "ON", "YES", "TRUE", "Y"))


def elf_id(app):
    require(app[:1] == b"\xe9" and app[32:36] == bytes.fromhex("3254cdab"), "Invalid ESP app descriptor")
    return app[176:208].hex()


def from_image(app):
    origins = set(re.findall(rb"http://[0-9]+(?:\.[0-9]+){3}(?::[0-9]+)?(?=\x00)", app))
    logins = set(re.findall(rb"http://[0-9]+(?:\.[0-9]+){3}(?::[0-9]+)?/login/agent-key(?=\x00)", app))

    def contains(value):
        return value.encode() + b"\x00" in app

    pairs = []
    for origin in origins:
        for login in logins:
            candidate = Services(origin.decode(), login.decode(), False)
            key = candidate.fingerprint
            if contains("na_" + key) and contains("pv_" + key):
                pairs.append(candidate)
    require(len(pairs) == 1, "Expected one development API/login pair matching both compiled NVS namespaces")
    candidate, = pairs
    key = candidate.fingerprint
    dev = all(contains(value) for value in ("nn_" + key, "ns_" + key,
              candidate.api.replace("http:", "ws:") + "/api/v1/nodes/ws",
              candidate.login.replace("/agent-key", "/device")))
    public = all(contains(value) for value in ("nyx", "nyx_setup", "wss://nyx-api.chrono-ai.fun/api/v1/nodes/ws",
                                                "https://nyxid.ai/login/device"))
    require(public != dev, "Ambiguous or incomplete node environment in app")
    return Services(candidate.api, candidate.login, dev)


def check_sources(board, playback_only=False):
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=board, text=True).strip()
    require(rev == check_builds.BOARD_COMMIT, "Unexpected board source commit")
    with TemporaryDirectory() as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        patches = (PATCH, PLAYBACK_ONLY) if playback_only else (PATCH,)
        commands = [["read-tree", rev], *(["apply", "--cached", str(p)] for p in patches), ["diff", "--quiet", "--"]]
        for args in commands:
            subprocess.run(["git", *args], cwd=board, env=env, check=True)


def historical_profile(app):
    digest = sha256(app).hexdigest()
    records = json.loads(MAC_RECORD.read_text())["historical_apps"]
    record = next((r for r in records if r["sha256"] == digest), None)
    require(record is not None, "Unrecognized historical rev1 app SHA256")
    descriptor = elf_id(app)
    require(descriptor == record["descriptor_elf_sha256"] and len(app) == record.get("bytes", len(app)),
            "Historical rev1 descriptor or length mismatch")
    services = from_image(app)
    require(services.fingerprint == record["service_fingerprint"], "Historical service fingerprint mismatch")
    if "node_environment" in record:
        require(services.node_dev == (record["node_environment"] == "development"), "Historical node mode mismatch")
    return services, dict(kind="historical-app", observation=record["observation"], sha256=digest,
                          bytes=len(app), descriptor_elf_sha256=descriptor)


def check_build(board, build, services, source, gdb, playback_only=False):
    desc = json.loads((build / "project_description.json").read_text())
    require(Path(desc["project_path"]).resolve() == board, "Build belongs to a different board checkout")
    app = (build / desc["app_bin"]).read_bytes()
    require((b"ref=physical tx=unavailable policy=fixed-playback-only" in app) == playback_only,
            "Linked app wake policy does not match the requested source profile")
    require(from_cache(build / "CMakeCache.txt") == services == from_image(app),
            "Source, effective cache and linked service profiles disagree")
    key = services.fingerprint
    commands = json.loads((build / "compile_commands.json").read_text())
    defines = dict(NYX_AGENT_BASE_URL=services.api, NYX_AGENT_VERIFICATION_URI=services.login,
                   NYX_AGENT_NVS_NAMESPACE="na_" + key, NYX_VOICE_NVS_NAMESPACE="pv_" + key)
    node = dict(HA_CAM_NYX_BASE_URL=services.api, HA_CAM_NYX_WSS_URL=services.api.replace("http:", "ws:") + "/api/v1/nodes/ws",
                NYX_NODE_VERIFICATION_URI=services.login.replace("/agent-key", "/device"),
                NYX_NODE_NVS_NAMESPACE="nn_" + key, NYX_NODE_SETUP_NVS_NAMESPACE="ns_" + key)
    for name in ("panel_voice.c", "nyx_node.c", "device_code.c", "nyx_creds.c"):
        entry, = [c for c in commands if Path(c["file"]).name == name]
        flags = shlex.split(entry["command"]) if "command" in entry else entry["arguments"]
        macros = dict(f[2:].split("=", 1) for f in flags if f.startswith("-D") and "=" in f)
        expected = defines | (node if services.node_dev else {})
        require(all(macros.get(k) == f'"{v}"' for k, v in expected.items()), "Service compile definition mismatch")
        if services.node_dev:
            require(macros.get("NYX_NODE_ALLOW_BAKED_CREDENTIALS") == "0", "Unexpected node credential mode")
        else:
            require(not (macros.keys() & (node.keys() | {"NYX_NODE_ALLOW_BAKED_CREDENTIALS"})), "Unexpected node overrides")
    record = check_builds.image(build, gdb)
    require(record["profile"] == PROFILE and elf_id(app) == record["elf_sha256"], "Board or ELF descriptor mismatch")
    return dict(source=source, service_fingerprint=key, node_environment="development" if services.node_dev else "public",
                board_commit=check_builds.BOARD_COMMIT, patch_sha256=sha256(PATCH.read_bytes()).hexdigest(),
                idf=desc["git_revision"], model_sha256=check_builds.MODEL, model_bytes=60952,
                threshold=0.91796875, smoothing_steps=5, descriptor_elf_sha256=elf_id(app), image=record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path, help="patched board checkout")
    p.add_argument("build", type=Path, help="new build directory (existing for --verify-only)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--old-app", type=Path, help="an archived rev1 app listed in the Mac handoff record")
    group.add_argument("--old-cache", type=Path, help="CMakeCache.txt retained from the known rev1 build")
    group.add_argument("--example", action="store_true", help="synthetic profile for a host build check")
    p.add_argument("--fixed-playback-only", action="store_true", help="verify the separate fixed-playback-only patch on the board sources")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--gdb", default="riscv32-esp-elf-gdb")
    args = p.parse_args()
    require(__debug__, "Use normal Python so that all linked-image checks run")
    board, build = args.board.resolve(), args.build.resolve()
    if args.old_app:
        services, source = historical_profile(args.old_app.read_bytes())
    elif args.old_cache:
        services = from_cache(args.old_cache)
        source = dict(kind="operator-selected-cache", sha256=sha256(args.old_cache.read_bytes()).hexdigest())
    else:
        services = Services("http://192.168.254.251:8100", "http://192.168.254.251:8200/login/agent-key", True)
        source = dict(kind="example-only")
    check_sources(board, args.fixed_playback_only)
    if not args.verify_only:
        build.mkdir(parents=True, exist_ok=True)
        require(not any(build.iterdir()), "Use an empty output directory for a new build")
        settings = dict(PANEL_BOARD_PROFILE=PROFILE, SDKCONFIG=str(build / "sdkconfig"), NYX_PANEL_WAKE_ENGINE="hey-nyx",
                        NYX_PANEL_RESEARCH_FIXED_MARKER="ON", **services.cmake())
        options = [word for k, v in settings.items() for word in ("-D", f"{k}={v}")]
        with (build / "dev-build.log").open("w") as log:
            for command in (["idf.py", "-B", build, *options, "reconfigure"], ["cmake", "--build", build, "--parallel", "2"]):
                code = subprocess.run(command, cwd=board, stdout=log, stderr=subprocess.STDOUT).returncode
                require(code == 0, "Build failed. Inspect dev-build.log locally")
    report = check_build(board, build, services, source, args.gdb, args.fixed_playback_only)
    if args.fixed_playback_only:
        report["playback_only_patch_sha256"] = sha256(PLAYBACK_ONLY.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
