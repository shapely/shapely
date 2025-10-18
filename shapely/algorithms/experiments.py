import time
import csv
from typing import List
from rand_rect_poly import generate_rectilinear_polygon
import experiments_csv as excsv
from shapely.geometry.linestring import LineString
import matplotlib.pyplot as plt
from min_partition_heuristic import partition_polygon as partition_polygon_huristic
from min_partition import partition_polygon as partition_polygon
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
            poly = generate_rectilinear_polygon(10)
            plotting(poly, [])

            # Test 'after' implementation
            start = time.time()
            partition_result_after = partition_polygon(poly)
            end = time.time()
            run_time_after = end - start
            partition_length_after = sum_edge_lengths(partition_result_after)
            print(f"Partition length after: {partition_length_after}")
            plotting(poly, partition_result_after)

            # Test 'before' implementation
            start = time.time()
            partition_result_before = partition_polygon_huristic(poly)
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
                    "polygon": poly,
                    "before_partition": partition_result_before,
                    "after_partition": partition_result_after,
                    "before_time": run_time_before,
                    "before_length": partition_length_before,
                    "after_time": run_time_after,
                    "after_length": partition_length_after,
                }
            )

            # Write the data to the CSV file
            writer.writerow({
                "trial": instance,
                "before_time": run_time_before,
                "before_length": partition_length_before,
                "after_time": run_time_after,
                "after_length": partition_length_after,
            })

    return results


def sum_edge_lengths(partition: List[LineString]) -> float:
    return sum([line.length for line in partition])


def plot_partition_comparison(poly, before_partition, after_partition, trial):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the 'before' partition
    axes[0].set_title(f"Trial {trial} - Before Partition")
    axes[0].plot(*poly.exterior.xy, color="black")
    for line in before_partition:
        axes[0].plot(*line.xy, color="red") # Plot the inner partition

    # Plot the 'after' partition
    axes[1].set_title(f"Trial {trial} - After Partition")
    axes[1].plot(*poly.exterior.xy, color="black")
    for line in after_partition:
        axes[1].plot(*line.xy, color="green") # Plot the inner partition

    plt.tight_layout()
    plt.savefig(f"comparison_trial_{trial}.png")
    plt.close()

def plot_comparison(results):
    for result in results:
        trial = result["trial"]
        poly = result["polygon"]
        before_partition = result["before_partition"]
        after_partition = result["after_partition"]

        plot_partition_comparison(poly, before_partition, after_partition, trial)


if __name__ == "__main__":
    results = run_comparison_experiment(1, "./algorithms/comparison_results.csv")
    plot_comparison(results)
    print(
        "Experiment completed. Results saved in 'comparison_results.csv' and 'comparison_trial_{trial}.png'."
    )
