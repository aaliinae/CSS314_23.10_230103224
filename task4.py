
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import numpy as np

STUDENT_ID = 230103224
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3


def queue_worker(input_queue, result_queue):
    """Receive a pickled array through IPC and calculate a checksum."""
    array = input_queue.get()
    result_queue.put(int(array.sum(dtype=np.int64)))


def shared_memory_worker(shm_name, size, result_connection):
    """Attach to existing shared memory without copying the array."""
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        array = np.ndarray((size,), dtype=np.int64, buffer=shm.buf)
        result_connection.send(int(array.sum(dtype=np.int64)))
    finally:
        shm.close()
        result_connection.close()


def benchmark_queue(array):
    """Measure process startup, IPC serialization and transfer overhead."""
    input_queue = mp.Queue()
    result_queue = mp.Queue()

    process = mp.Process(
        target=queue_worker,
        args=(input_queue, result_queue)
    )

    start = time.perf_counter()
    process.start()
    input_queue.put(array)

    result = result_queue.get()
    process.join()
    elapsed_ms = (time.perf_counter() - start) * 1000

    input_queue.close()
    result_queue.close()

    if process.exitcode != 0:
        raise RuntimeError("Queue worker failed")

    return elapsed_ms, result


def benchmark_shared_memory(array):
    """Measure process startup and shared-memory access overhead."""
    shm = shared_memory.SharedMemory(
        create=True,
        size=array.nbytes
    )

    try:
        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf
        )
        shared_array[:] = array

        # Exclude initial allocation and population from the timed transfer.
        parent_connection, child_connection = mp.Pipe(duplex=False)

        process = mp.Process(
            target=shared_memory_worker,
            args=(shm.name, len(array), child_connection)
        )

        start = time.perf_counter()
        process.start()
        child_connection.close()

        result = parent_connection.recv()
        process.join()
        elapsed_ms = (time.perf_counter() - start) * 1000

        parent_connection.close()

        if process.exitcode != 0:
            raise RuntimeError("Shared-memory worker failed")

        return elapsed_ms, result

    finally:
        shm.close()
        shm.unlink()


def median_benchmark(benchmark_function, array):
    timings = []
    checksums = []

    for _ in range(REPEATS):
        elapsed_ms, checksum = benchmark_function(array)
        timings.append(elapsed_ms)
        checksums.append(checksum)

    if len(set(checksums)) != 1:
        raise RuntimeError("Inconsistent checksums across repetitions")

    return statistics.median(timings), checksums[0]


def main():
    rng = np.random.default_rng(STUDENT_ID)
    results = []

    print("=" * 78)
    print("TASK 4: IPC SERIALIZATION VS SHARED MEMORY")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Repeats per path: {REPEATS} (median reported)")
    print("=" * 78)

    for size in SIZES:
        print(f"\nBenchmarking N = {size:,} ...")

        array = rng.integers(
            1, 11, size=size, dtype=np.int64
        )

        queue_ms, queue_sum = median_benchmark(
            benchmark_queue, array
        )
        shared_ms, shared_sum = median_benchmark(
            benchmark_shared_memory, array
        )

        if queue_sum != shared_sum:
            raise AssertionError(
                f"Checksum mismatch for N={size}"
            )

        results.append({
            "size": size,
            "queue_ms": queue_ms,
            "shared_ms": shared_ms,
            "checksum": queue_sum
        })

    lines = [
        "\n=== TASK 4 RESULTS ===",
        f"STUDENT_ID: {STUDENT_ID}",
        f"REPEATS: {REPEATS}",
        "Times are median wall-clock milliseconds.",
        "Both paths include worker process startup.",
        "Shared-memory creation/population is excluded from its timed interval.",
        "",
        f"{'N':>12} | {'IPC Queue (ms)':>16} | "
        f"{'Shared Memory (ms)':>19} | {'Checksum':>16}",
        "-" * 75,
    ]

    for result in results:
        lines.append(
            f"{result['size']:>12,} | "
            f"{result['queue_ms']:>16.3f} | "
            f"{result['shared_ms']:>19.3f} | "
            f"{result['checksum']:>16}"
        )

    output = "\n".join(lines) + "\n"
    print(output)

    with open("task4_output.txt", "w", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    mp.freeze_support()
    main()
