import time
import csv
from typing import List
from rand_rect_poly import generate_rectilinear_polygon
import experiments_csv as excsv
from shapely.geometry.linestring import LineString
import matplotlib.pyplot as plt
from min_partition_before import partition_polygon as partition_polygon_before
from min_partition_after import partition_polygon as partition_polygon_after
from plot_poly import plotting


def run_comparison_experiment(num_trials: int, output_file: str):
    """
    Run a series of experiments to compare partition_polygon implementations.

    Args:
        num_trials (int): The number of experiments to run.
        output_file (str): The CSV file to log the results.
    """
    logger = excsv.logger

    results = []

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "trial",
            "before_time",
            "before_length",
            "after_time",
            "after_length",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in range(num_trials):
            poly = generate_rectilinear_polygon()
            plotting(poly, [])

            # Test 'after' implementation
            start = time.time()
            partition_result_after = partition_polygon_after(poly)
            end = time.time()
            run_time_after = end - start
            partition_length_after = sum_edge_lengths(partition_result_after)
            logger.warning(f"Partition length after: {partition_length_after}")
            plotting(poly, partition_result_after)

            # Test 'before' implementation
            start = time.time()
            partition_result_before = partition_polygon_before(poly)
            end = time.time()
            run_time_before = end - start
            partition_length_before = sum_edge_lengths(partition_result_before)

            # Log the data
            logger.info(f"Trial {instance}:")
            logger.info(
                f"Before: Runtime = {run_time_before}, Length = {partition_length_before}"
            )
            logger.info(
                f"After: Runtime = {run_time_after}, Length = {partition_length_after}"
            )

            # Store results for graphing
            results.append(
                {
                    "trial": instance,
                    "before_time": run_time_before,
                    "before_length": partition_length_before,
                    "after_time": run_time_after,
                    "after_length": partition_length_after,
                }
            )

            # Write the data to the CSV file
            writer.writerow(results[-1])

    return results


def sum_edge_lengths(partition: List[LineString]) -> float:
    return sum([line.length for line in partition])


def plot_comparison(results):
    trials = [r["trial"] for r in results]
    before_times = [r["before_time"] for r in results]
    after_times = [r["after_time"] for r in results]
    before_lengths = [r["before_length"] for r in results]
    after_lengths = [r["after_length"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Runtime comparison
    ax1.plot(trials, before_times, label="Before", marker="o")
    ax1.plot(trials, after_times, label="After", marker="s")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title("Runtime Comparison")
    ax1.legend()

    # Partition length comparison
    ax2.plot(trials, before_lengths, label="Before", marker="o")
    ax2.plot(trials, after_lengths, label="After", marker="s")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Partition Length")
    ax2.set_title("Partition Length Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = run_comparison_experiment(1, "comparison_results.csv")
    plot_comparison(results)
    print(
        "Experiment completed. Results saved in 'comparison_results.csv' and 'performance_comparison.png'."
    )
