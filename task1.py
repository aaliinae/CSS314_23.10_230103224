import os
import time
import platform
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103224
MATRIX_SIZE = 8000


def get_cpu_count(command):
    import subprocess

    try:
        result = subprocess.check_output(
            ["sysctl", "-n", command],
            text=True
        ).strip()
        return int(result)
    except Exception:
        return os.cpu_count() or 1


def initialize_matrix(shm_name, shape, dtype, seed):
    shm = shared_memory.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.random.seed(seed)
    matrix[:] = np.random.random(shape)
    shm.close()


def sum_rows(args):
    shm_name, shape, dtype, start, end = args
    shm = shared_memory.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    result = float(matrix[start:end, :].sum(dtype=np.float64))

    shm.close()
    return result


def sum_columns(args):
    shm_name, shape, dtype, start, end = args
    shm = shared_memory.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Columns are processed across all rows.
    # C-order storage makes this access strided.
    result = float(matrix[:, start:end].sum(dtype=np.float64))

    shm.close()
    return result


def parallel_sum(pool, worker_function, shm_name, shape,
                 dtype, workers, axis_size):
    chunk_size = (axis_size + workers - 1) // workers

    tasks = [
        (
            shm_name,
            shape,
            dtype,
            start,
            min(start + chunk_size, axis_size)
        )
        for start in range(0, axis_size, chunk_size)
    ]

    partial_sums = pool.map(worker_function, tasks)
    return sum(partial_sums)


def main():
    logical_cores = get_cpu_count("hw.logicalcpu")
    physical_cores = get_cpu_count("hw.physicalcpu")

    # Use physical cores when available.
    workers = max(1, physical_cores)

    shape = (MATRIX_SIZE, MATRIX_SIZE)
    dtype = np.dtype(np.float64)
    matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * dtype.itemsize

    print("=" * 55)
    print("TASK 1: ROW-MAJOR VS COLUMN-MAJOR CACHE PROFILING")
    print(f"Student ID       : {STUDENT_ID}")
    print(f"Logical cores    : {logical_cores}")
    print(f"Physical cores   : {physical_cores}")
    print(f"Workers          : {workers}")
    print(f"Matrix           : {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"Memory           : {matrix_bytes / (1024**2):.2f} MiB")
    print("=" * 55)

    shm = shared_memory.SharedMemory(
        create=True, size=matrix_bytes
    )

    try:
        matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Seed the matrix using the student ID.
        rng = np.random.default_rng(STUDENT_ID)
        matrix[:] = rng.random(shape, dtype=np.float64)

        del matrix

        with mp.Pool(processes=workers) as pool:
            # Mode A: workers process contiguous row blocks.
            start = time.perf_counter()
            sum_a = parallel_sum(
                pool, sum_rows, shm.name, shape,
                dtype, workers, MATRIX_SIZE
            )
            time_a = time.perf_counter() - start

            # Mode B: workers process column blocks.
            start = time.perf_counter()
            sum_b = parallel_sum(
                pool, sum_columns, shm.name, shape,
                dtype, workers, MATRIX_SIZE
            )
            time_b = time.perf_counter() - start

        penalty = time_b / time_a if time_a > 0 else float("nan")

        output = (
            "\n"
            "=== TASK 1 RESULTS ===\n"
            f"STUDENT_ID       : {STUDENT_ID}\n"
            f"LOGICAL_CORES    : {logical_cores}\n"
            f"PHYSICAL_CORES   : {physical_cores}\n"
            f"WORKERS          : {workers}\n"
            f"MATRIX_SIZE      : {MATRIX_SIZE} x {MATRIX_SIZE}\n"
            f"MODE_A_ROW_MS    : {time_a * 1000:.3f}\n"
            f"MODE_B_COLUMN_MS : {time_b * 1000:.3f}\n"
            f"CACHE_PENALTY    : {penalty:.4f}x\n"
            f"SUM_A            : {sum_a:.10f}\n"
            f"SUM_B            : {sum_b:.10f}\n"
            f"SUM_MATCH        : {np.isclose(sum_a, sum_b, rtol=1e-10)}\n"
        )

        print(output)

        with open("task1_output.txt", "w", encoding="utf-8") as file:
            file.write(output)

        if not np.isclose(sum_a, sum_b, rtol=1e-10):
            raise RuntimeError("The two sums do not match!")

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
