"""Phase 4 End-to-End Node.js Proxy -> FastAPI GPU Backend Verification Script.

Spawns backend/server.py and frontend_node/server.js as background processes,
polls http://127.0.0.1:3000/api/health until ready,
sends multipart image to http://127.0.0.1:3000/api/zombify,
asserts JPEG magic bytes (0xFFD8) and zero disk writes,
cleanly shuts down all processes, and logs results.
"""

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def list_files_in_tree(root_dir: Path) -> set:
    """Return set of relative file paths in root_dir, ignoring log and cache files."""
    files = set()
    for root, _, filenames in os.walk(root_dir):
        rel_root = Path(root).relative_to(root_dir)
        # Skip logs and node_modules from strict check
        if any(
            part in ("logs", "node_modules", "__pycache__", ".git")
            for part in rel_root.parts
        ):
            continue
        for f in filenames:
            rel_file = rel_root / f
            files.add(str(rel_file))
    return files


def run_e2e_verification():
    base_dir = Path("photobooth_v2").resolve()
    if not base_dir.exists():
        base_dir = Path(".").resolve()

    backend_dir = base_dir / "backend"
    frontend_node_dir = base_dir / "frontend_node"
    artifacts_dir = base_dir / "artifacts"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    test_input = artifacts_dir / "test_input_640x480.jpg"
    assert test_input.exists(), f"Missing input artifact: {test_input}"

    backend_stdout_log = logs_dir / "backend_e2e_stdout.log"
    node_stdout_log = logs_dir / "node_e2e_stdout.log"
    phase4_log_file = logs_dir / "phase4_e2e_verification.log"

    log_records = [
        f"[{datetime.datetime.now().isoformat()}] Starting Phase 4 E2E Integration Verification",
        f"Base Dir: {base_dir}",
        f"Python executable: {sys.executable}",
    ]

    # Snapshot disk files before test
    files_before = list_files_in_tree(base_dir)

    backend_proc = None
    node_proc = None

    try:
        log_records.append("Launching FastAPI GPU Backend on port 8000...")
        b_out = open(backend_stdout_log, "w", encoding="utf-8")
        backend_proc = subprocess.Popen(
            [sys.executable, str(backend_dir / "server.py")],
            cwd=str(base_dir),
            stdout=b_out,
            stderr=subprocess.STDOUT,
        )

        log_records.append("Launching Node.js Express Kiosk on port 3000...")
        n_out = open(node_stdout_log, "w", encoding="utf-8")
        node_env = os.environ.copy()
        node_env["PORT"] = "3000"
        node_env["REMOTE_GPU_URL"] = "http://127.0.0.1:8000"

        node_proc = subprocess.Popen(
            ["node", "server.js"],
            cwd=str(frontend_node_dir),
            env=node_env,
            stdout=n_out,
            stderr=subprocess.STDOUT,
        )

        # Poll http://127.0.0.1:3000/api/health until ready
        health_url = "http://127.0.0.1:3000/api/health"
        log_records.append(
            f"Polling {health_url} until proxy and backend report ready..."
        )

        poll_timeout = 150.0  # Allow model initialization and warmup
        poll_start = time.perf_counter()
        ready = False
        health_data = {}

        while time.perf_counter() - poll_start < poll_timeout:
            # Check if backend died early
            if backend_proc.poll() is not None:
                raise RuntimeError(
                    f"Backend process terminated prematurely with code {backend_proc.returncode}. Check {backend_stdout_log}"
                )
            if node_proc.poll() is not None:
                raise RuntimeError(
                    f"Node.js process terminated prematurely with code {node_proc.returncode}. Check {node_stdout_log}"
                )

            try:
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    health_data = resp.json()
                    if (
                        health_data.get("status") == "ready"
                        and health_data.get("node_proxy") == "ok"
                    ):
                        ready = True
                        log_records.append(
                            f"Proxy & GPU backend reported ready in {time.perf_counter() - poll_start:.1f}s"
                        )
                        log_records.append(f"Health Response: {health_data}")
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise TimeoutError(
                f"Health check timed out after {poll_timeout}s. Last data: {health_data}"
            )

        # Send automated multipart POST request to Node.js proxy
        zombify_url = "http://127.0.0.1:3000/api/zombify"
        log_records.append(
            f"Sending multipart POST with {test_input.name} to {zombify_url}..."
        )

        t_req_start = time.perf_counter()
        with open(test_input, "rb") as f:
            files = {"file": ("test_input_640x480.jpg", f, "image/jpeg")}
            proxy_resp = requests.post(zombify_url, files=files, timeout=60.0)
        roundtrip_ms = (time.perf_counter() - t_req_start) * 1000.0

        log_records.append(f"HTTP Response Code: {proxy_resp.status_code}")
        log_records.append(f"Total Roundtrip Proxy Latency: {roundtrip_ms:.2f} ms")
        log_records.append(
            f"Upstream X-Gen-Time Header: {proxy_resp.headers.get('X-Gen-Time', 'N/A')}"
        )
        log_records.append(f"Content-Type: {proxy_resp.headers.get('Content-Type')}")
        log_records.append(f"Response Payload Size: {len(proxy_resp.content)} bytes")

        assert proxy_resp.status_code == 200, (
            f"Expected 200 OK, got {proxy_resp.status_code}: {proxy_resp.text}"
        )
        assert len(proxy_resp.content) > 5000, (
            f"Payload too small: {len(proxy_resp.content)} bytes"
        )

        # Verify JPEG magic bytes
        # 0xFF 0xD8 (SOI - Start of Image) and 0xFF 0xD9 (EOI - End of Image)
        is_jpeg = proxy_resp.content.startswith(
            b"\xff\xd8"
        ) and proxy_resp.content.endswith(b"\xff\xd9")
        log_records.append(f"JPEG Magic Bytes (0xFFD8 ... 0xFFD9) Verified: {is_jpeg}")
        assert is_jpeg, (
            f"Response payload is not a valid JPEG! Header bytes: {proxy_resp.content[:4].hex()}"
        )

        # Snapshot disk files after test to verify zero disk writes
        files_after = list_files_in_tree(base_dir)
        new_files = files_after - files_before
        log_records.append(
            f"New non-log disk files created during inference: {list(new_files)}"
        )
        assert len(new_files) == 0, (
            f"Local disk writes detected during streaming! New files: {new_files}"
        )
        log_records.append(
            "Zero Local Disk Writes Assertion: PASSED (Pure in-memory RAM streaming)"
        )

        log_records.append("E2E_PROXY_STATUS: PASS")
        print("\n".join(log_records))

    finally:
        # Clean shutdown of both processes
        log_records.append("Terminating background test processes...")
        if node_proc and node_proc.poll() is None:
            node_proc.terminate()
            try:
                node_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                node_proc.kill()
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()

        log_records.append("Background processes terminated cleanly.")

        # Write complete log
        with open(phase4_log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(log_records) + "\n")


if __name__ == "__main__":
    run_e2e_verification()
