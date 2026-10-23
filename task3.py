
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103224
ARRAY_SIZE = 2_000_000
THREADS = 4


def serial_prefix_sum(data):
    """Serial inclusive prefix sum."""
    output = np.empty_like(data)
    np.cumsum(data, dtype=np.int64, out=output)
    return output


def compute_local_scan(args):
    """Compute the inclusive prefix sum and total of one chunk."""
    chunk, start, end = args
    local_output = np.cumsum(chunk[start:end], dtype=np.int64)
    chunk_sum = int(local_output[-1]) if len(local_output) else 0
    return start, end, local_output, chunk_sum


def apply_offset(args):
    """Add the preceding chunks' total to a local prefix sum."""
    start, end, local_output, offset = args
    if offset:
        local_output += offset
    return start, end, local_output


def parallel_prefix_sum(data, workers):
    """Two-pass parallel chunked inclusive prefix scan."""
    n = len(data)
    boundaries = np.linspace(0, n, workers + 1, dtype=int)

    chunks = [
        (data, int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(workers)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Pass 1: local prefix scans and local chunk sums.
        local_results = list(executor.map(compute_local_scan, chunks))

        # Offset reduction: exclusive prefix of chunk totals.
        chunk_sums = [result[3] for result in local_results]
        offsets = [0] * workers
        running_total = 0

        for i, chunk_sum in enumerate(chunk_sums):
            offsets[i] = running_total
            running_total += chunk_sum

        # Pass 2: add the preceding chunks' totals.
        offset_tasks = [
            (
                start,
                end,
                local_output,
                offsets[i]
            )
            for i, (start, end, local_output, _) in enumerate(local_results)
        ]

        adjusted_results = list(executor.map(apply_offset, offset_tasks))

    output = np.empty_like(data)

    for start, end, local_output in adjusted_results:
        output[start:end] = local_output

    return output


def main():
    workers = min(THREADS, os.cpu_count() or 1)

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    print("=" * 60)
    print("TASK 3: PREFIX SCAN")
    print(f"STUDENT_ID : {STUDENT_ID}")
    print(f"ARRAY_SIZE : {ARRAY_SIZE:,}")
    print(f"THREADS    : {workers}")
    print("=" * 60)

    start = time.perf_counter()
    serial_output = serial_prefix_sum(data)
    serial_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    parallel_output = parallel_prefix_sum(data, workers)
    parallel_ms = (time.perf_counter() - start) * 1000

    verification = (
        int(parallel_output[-1]) == int(serial_output[-1])
    )

    output_text = (
        "\n=== TASK 3 RESULTS ===\n"
        f"STUDENT_ID       : {STUDENT_ID}\n"
        f"ARRAY_SIZE       : {ARRAY_SIZE}\n"
        f"THREADS          : {workers}\n"
        f"SERIAL_TIME_MS   : {serial_ms:.3f}\n"
        f"PARALLEL_TIME_MS : {parallel_ms:.3f}\n"
        f"SPEEDUP          : {serial_ms / parallel_ms:.4f}x\n"
        f"SERIAL_LAST      : {int(serial_output[-1])}\n"
        f"PARALLEL_LAST    : {int(parallel_output[-1])}\n"
        f"VERIFICATION     : {'PASS' if verification else 'FAIL'}\n"
    )

    print(output_text)

    with open("task3_output.txt", "w", encoding="utf-8") as file:
        file.write(output_text)

    if not verification:
        print("ERROR: Prefix scan verification failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()