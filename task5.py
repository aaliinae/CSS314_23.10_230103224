
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103224
TASK_COUNT = 10_000
HASH_ITERATIONS = 500
PAYLOAD = b"A" * 1024


def cpu_bound_task(task_id):
    """Hash a 1 KB payload 500 times."""
    data = PAYLOAD + task_id.to_bytes(4, "little")

    for _ in range(HASH_ITERATIONS):
        data = hashlib.sha256(data).digest()

    return data[0]


def run_benchmark(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(cpu_bound_task, range(TASK_COUNT)))

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Consume results to ensure all tasks completed successfully.
    checksum = sum(results)
    return elapsed_ms, checksum


def get_pool_sizes(cores):
    candidates = [
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256,
    ]

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(candidates))


def main():
    logical_cores = os.cpu_count() or 1
    pool_sizes = get_pool_sizes(logical_cores)

    print("=" * 86)
    print("TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD")
    print(f"Student ID       : {STUDENT_ID}")
    print(f"Logical cores    : {logical_cores}")
    print(f"CPU-bound tasks  : {TASK_COUNT:,}")
    print(f"Hashes per task  : {HASH_ITERATIONS}")
    print(f"Payload size     : 1 KB")
    print("=" * 86)

    results = []

    for pool_size in pool_sizes:
        print(f"Testing pool size {pool_size}...", flush=True)
        elapsed_ms, checksum = run_benchmark(pool_size)

        results.append({
            "pool_size": pool_size,
            "time_ms": elapsed_ms,
            "checksum": checksum,
        })

    baseline_ms = results[0]["time_ms"]
    first_worsening = None
    best_time_so_far = baseline_ms

    # Mark the first tested pool size that is slower than
    # the best runtime observed at any smaller tested pool size.
    for i, result in enumerate(results):
        result["speedup"] = baseline_ms / result["time_ms"]

        if i == 0:
            result["status"] = "BASELINE"
            continue

        if first_worsening is None and result["time_ms"] > best_time_so_far:
            first_worsening = result["pool_size"]
            result["status"] = "FIRST WORSENING"
        elif result["time_ms"] > best_time_so_far:
            result["status"] = "SLOWER THAN BEST"
        else:
            result["status"] = "NO NEW SLOWDOWN"

        best_time_so_far = min(best_time_so_far, result["time_ms"])

    lines = [
        "\n=== TASK 5 RESULTS ===",
        f"STUDENT_ID: {STUDENT_ID}",
        f"LOGICAL_CORES: {logical_cores}",
        f"TASK_COUNT: {TASK_COUNT}",
        f"HASH_ITERATIONS_PER_TASK: {HASH_ITERATIONS}",
        "",
        f"{'Pool Size':>10} | {'Execution Time (ms)':>19} | "
        f"{'Speedup':>10} | {'Status'}",
        "-" * 75,
    ]

    for result in results:
        lines.append(
            f"{result['pool_size']:>10} | "
            f"{result['time_ms']:>19.3f} | "
            f"{result['speedup']:>9.3f}x | "
            f"{result['status']}"
        )

    lines.append("")

    if first_worsening is None:
        lines.append(
            "FIRST WORSENING POOL SIZE: Not observed in tested pool sizes"
        )
    else:
        lines.append(
            f"FIRST WORSENING POOL SIZE: {first_worsening}"
        )

    lines.append(
        "Note: The first slowdown is an observed performance threshold; "
        "timing alone cannot prove context switching is its sole cause."
    )

    output = "\n".join(lines) + "\n"
    print(output)

    with open("task5_output.txt", "w", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    main()
