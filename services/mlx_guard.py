"""Memory guard for MLX jobs on the shared Mac Studio (training and evaluation)."""

import os
import subprocess  # nosec B404
import threading
import time

MIN_FREE_PERCENT = 20  # stop the job if the whole Mac falls below this much free memory


def _system_free_percent() -> int | None:
    """macOS's own free-memory level (0-100), as memory_pressure reports it."""
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def memory_guard(max_gb: float, min_free_percent: int = MIN_FREE_PERCENT) -> None:
    """Stop the process if MLX memory (active plus cache) passes max_gb, or if the whole Mac drops
    below min_free_percent free. The Mac is shared and has no swap, and a model that does not fit
    (Qwen3.6-27B reached 85-91 GB) must fail cleanly instead of starving other users. Used by
    training and, through services.llm, by every MLX evaluation."""
    import mlx.core as mx

    def stop(reason: str) -> None:
        print(f"memory guard: {reason}, stopping", flush=True)
        os._exit(3)

    def watch() -> None:
        tick = 0
        while True:
            used = (mx.get_active_memory() + mx.get_cache_memory()) / 1e9
            if used > max_gb:
                stop(f"MLX {used:.1f} GB > {max_gb} GB")
            if tick % 10 == 0:  # the system check is slower, once a second
                free = _system_free_percent()
                if free is not None and free < min_free_percent:
                    stop(f"system free memory {free}% < {min_free_percent}%")
            tick += 1
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True).start()
