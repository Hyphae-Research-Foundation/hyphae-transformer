"""Hard process boundary for one-shot knowledge workers."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import IO, cast


@dataclass(frozen=True, slots=True)
class SupervisedResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool


def run_supervised(
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
    grace_seconds: float = 2.0,
    maximum_output_bytes: int = 1_000_000,
    input_bytes: bytes | None = None,
) -> SupervisedResult:
    if (
        not command
        or timeout_seconds <= 0
        or grace_seconds <= 0
        or len(input_bytes or b"") > 1_000_000
    ):
        raise ValueError("supervised process limits are invalid")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL if input_bytes is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        close_fds=True,
        start_new_session=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert process.stdout is not None and process.stderr is not None
    if process.stdin is not None:
        os.set_blocking(process.stdin.fileno(), False)
    for stream in (process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending_input = memoryview(input_bytes or b"")
    if process.stdin is not None:
        if pending_input:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    try:
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                stream = cast(IO[bytes], key.fileobj)
                if key.data == "stdin":
                    try:
                        written = os.write(stream.fileno(), pending_input)
                    except BrokenPipeError:
                        written = len(pending_input)
                    pending_input = pending_input[written:]
                    if not pending_input:
                        selector.unregister(stream)
                        stream.close()
                    continue
                chunk = os.read(stream.fileno(), 65_536)
                if not chunk:
                    selector.unregister(stream)
                    continue
                output = outputs[key.data]
                output.extend(chunk)
                if len(output) > maximum_output_bytes:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError("supervised process output exceeds its byte bound")
        if timed_out:
            try:
                process.wait(timeout=min(grace_seconds, 0.1))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        else:
            process.wait()
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            while True:
                try:
                    chunk = os.read(stream.fileno(), 65_536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                outputs[name].extend(chunk)
                if len(outputs[name]) > maximum_output_bytes:
                    raise RuntimeError("supervised process output exceeds its byte bound")
    finally:
        selector.close()
    return SupervisedResult(
        process.returncode,
        bytes(outputs["stdout"]),
        bytes(outputs["stderr"]),
        timed_out,
    )
