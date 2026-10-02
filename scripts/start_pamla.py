"""Save as scripts/start_pamla.py; run with your project Python environment active.

    python scripts/start_pamla.py

Optional environment variables: LLAMA_SERVER, GEMMA_MODEL_PATH.

- Verifies that ports 8000 and 8080 are available, then starts PAMLA on 8000 and Gemma on 8080.
"""

import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def stop_services(children, timeout=10):
    """Stop only the isolated process groups created by this launcher (macOS)."""
    def signal_group(process, sig):
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False

    for process in reversed(children):
        signal_group(process, signal.SIGTERM)
    for process in reversed(children):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            process.wait()



def main():
    children = []
    server = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if not server and Path("/opt/homebrew/bin/llama-server").is_file():
        server = "/opt/homebrew/bin/llama-server"
    model = Path(os.environ.get(
        "GEMMA_MODEL_PATH", "models/gemma-3-4b-it-q4_0.gguf"
    )).expanduser()
    if not model.is_absolute():
        model = ROOT / model

    if not server:
        raise RuntimeError("Install llama.cpp or set LLAMA_SERVER to its executable path.")
    if not model.is_file():
        raise RuntimeError(f"Download Gemma to {model} before starting PAMLA.")
    if not (ROOT / "src" / "app.py").is_file():
        raise RuntimeError("Place this launcher in your project's scripts/ folder.")

    # Fail before launching if either port is already occupied.
    for port in (8000, 8080):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"Port {port} is unavailable. Stop the existing service first."
                ) from error

    def stop_requested(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        print("Starting Gemma on http://127.0.0.1:8080 ...", flush=True)
        gemma = subprocess.Popen([
            server, "-m", str(model), "--alias", "pamla-gemma",
            "--ctx-size", "4096", "--host", "127.0.0.1", "--port", "8080"
        ], cwd=ROOT, start_new_session=True)
        children.append(gemma)

        deadline = time.monotonic() + 180
        while True:
            if gemma.poll() is not None:
                raise RuntimeError("Gemma exited during startup; see its output above.")
            try:
                with urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Gemma did not become ready within 180 seconds.")
            time.sleep(1)

        print("Gemma ready. Starting PAMLA at http://127.0.0.1:8000/docs", flush=True)
        api = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "src",
            "--host", "127.0.0.1", "--port", "8000",
        ], cwd=ROOT, start_new_session=True)
        children.append(api)
        print("Press Ctrl+C to stop both services.", flush=True)
        while True:
            for name, process in (("Gemma", gemma), ("FastAPI", api)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with code {process.returncode}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping PAMLA and Gemma...", flush=True)
    finally:
        # Ignore further interrupts while cleaning up only processes we started.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_services(children)



if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        sys.exit(1)
