import logging
import time
import csv
from typing import List
from rand_rect_poly import generate_rectilinear_polygon
from min_partition import partition_polygon
from plot_poly import plotting
import experiments_csv as excsv
from shapely.geometry.linestring import LineString

def run_experiment(instances: int, output_file: str):
    """
    Run a series of experiments to partition random polygons and log the results to a CSV file.

    Args:
        instances (int): The number of instances to run.
        output_file (str): The CSV file to log the results.
    """
    # Create a logger using the existing function in experiments_csv
    logger = excsv.logger

    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['trial', 'run_time', 'partition_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        for instance in range(instances):
            poly = generate_rectilinear_polygon()
            plotting(poly, [])

            start = time.time()
            partition_result = partition_polygon(poly)
            end = time.time()

            run_time = end - start
            partition_length = sum_edge_lengths(partition_result)

            # Log the data
            logger.info(f"instance {instance}: Runtime = {run_time}, Partition Length = {partition_length}")

            # Write the data to the CSV file
            writer.writerow({
                'instance': instance,
                'run_time': run_time,
                'partition_length': partition_length
            })

            plotting(poly, partition_result)  # Optionally plot the results for each instance.

def sum_edge_lengths(partition: List[LineString]) -> float:
    """
    Calculate the sum of the lengths of the edges in a partition.

    Args:
        partition (List[LineString]): A list of LineString objects 

    Returns:
        float: The sum of the lengths of the edges in the partition.
    """    
    return sum([line.length for line in partition])

if __name__ == "__main__":
    run_experiment(2, "experiment_results.csv")