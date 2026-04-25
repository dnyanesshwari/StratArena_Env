from __future__ import annotations

from evaluation.benchmarks import benchmark_all_tasks


def main() -> None:
    for row in benchmark_all_tasks():
        print(row)


if __name__ == "__main__":
    main()
