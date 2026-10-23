
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103224
TOTAL_POINTS = 40_000_000
THREADS = 4
CHUNK_SIZE = 250_000


def count_inside(x, y):
    """Count points inside the quarter-circle."""
    return int(np.count_nonzero(x * x + y * y <= 1.0))


def shared_prng_worker(thread_index, points_per_thread, rng, lock):
    """Use one shared PRNG protected by a mutex."""
    inside = 0
    remaining = points_per_thread

    while remaining > 0:
        count = min(CHUNK_SIZE, remaining)

        # Protect shared PRNG state while generating coordinates.
        with lock:
            x = rng.random(count)
            y = rng.random(count)

        # Calculation takes place outside the critical section.
        inside += count_inside(x, y)
        remaining -= count

    return inside


def independent_prng_worker(thread_index, points_per_thread):
    """Use an independent PRNG for each thread."""
    seed = (STUDENT_ID * 10) + thread_index
    rng = np.random.default_rng(seed)

    inside = 0
    remaining = points_per_thread

    while remaining > 0:
        count = min(CHUNK_SIZE, remaining)

        x = rng.random(count)
        y = rng.random(count)

        inside += count_inside(x, y)
        remaining -= count

    return inside


def run_shared_prng(points_per_thread):
    # One generator shared by all four threads.
    rng = np.random.default_rng(STUDENT_ID)
    lock = threading.Lock()

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(
                shared_prng_worker,
                i,
                points_per_thread,
                rng,
                lock
            )
            for i in range(THREADS)
        ]
        inside = sum(f.result() for f in futures)

    elapsed = time.perf_counter() - start
    pi_estimate = 4.0 * inside / TOTAL_POINTS
    error = abs(pi_estimate - math.pi)

    return pi_estimate, error, elapsed * 1000


def run_independent_prng(points_per_thread):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(
                independent_prng_worker,
                i,
                points_per_thread
            )
            for i in range(THREADS)
        ]
        inside = sum(f.result() for f in futures)

    elapsed = time.perf_counter() - start
    pi_estimate = 4.0 * inside / TOTAL_POINTS
    error = abs(pi_estimate - math.pi)

    return pi_estimate, error, elapsed * 1000


def main():
    points_per_thread = TOTAL_POINTS // THREADS

    if TOTAL_POINTS % THREADS != 0:
        raise ValueError("Total points must be divisible by thread count.")

    print("=" * 60)
    print("TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK")
    print(f"STUDENT_ID       : {STUDENT_ID}")
    print(f"TOTAL_POINTS     : {TOTAL_POINTS:,}")
    print(f"THREADS          : {THREADS}")
    print(f"POINTS_PER_THREAD: {points_per_thread:,}")
    print("=" * 60)

    print("\nRunning shared PRNG with mutex...")
    shared_pi, shared_error, shared_ms = run_shared_prng(
        points_per_thread
    )

    print("Running independent PRNGs...")
    independent_pi, independent_error, independent_ms = (
        run_independent_prng(points_per_thread)
    )

    speedup = (
        shared_ms / independent_ms
        if independent_ms > 0
        else float("nan")
    )

    output = (
        "\n=== TASK 2 RESULTS ===\n"
        f"STUDENT_ID          : {STUDENT_ID}\n"
        f"TOTAL_POINTS        : {TOTAL_POINTS}\n"
        f"THREADS             : {THREADS}\n"
        f"SHARED_PI           : {shared_pi:.10f}\n"
        f"SHARED_ABS_ERROR    : {shared_error:.10f}\n"
        f"SHARED_TIME_MS      : {shared_ms:.3f}\n"
        f"INDEPENDENT_PI      : {independent_pi:.10f}\n"
        f"INDEPENDENT_ABS_ERROR: {independent_error:.10f}\n"
        f"INDEPENDENT_TIME_MS : {independent_ms:.3f}\n"
        f"SPEEDUP             : {speedup:.4f}x\n"
    )

    print(output)

    with open("task2_output.txt", "w", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    main()
