"""Validate geometries and make them valid."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import partial

import shapely

__all__ = ["explain_validity", "make_valid", "make_valid_parallel"]


def explain_validity(ob):
    """Explain the validity of the input geometry, if it is invalid."""
    return shapely.is_valid_reason(ob)


def _make_valid_single(ob):
    """Internal single-geometry validation function"""
    if ob.is_valid:
        return ob
    return shapely.make_valid(ob)


def make_valid(ob, n_workers=None):
    """Make the input geometry valid with optional parallel processing.
    
    Parameters
    ----------
    ob : Geometry or array-like
        A single shapely geometry object or array-like of geometries
    n_workers : int, optional
        Number of parallel workers to use (None for single-threaded)
        
    Returns
    -------
    Geometry or array
        Validated geometry/geometries
    """
    if n_workers is None or n_workers == 1:
        if hasattr(ob, '__iter__'):
            return np.array([_make_valid_single(g) for g in ob])
        return _make_valid_single(ob)
    else:
        return make_valid_parallel(ob, n_workers)


def make_valid_parallel(geoms, n_workers=8):
    """Parallel implementation of make_valid for array-like geometries.
    
    Parameters
    ----------
    geoms : array-like
        Array of shapely geometries
    n_workers : int
        Number of parallel workers to use
        
    Returns
    -------
    np.ndarray
        Array of validated geometries
    """
    # Split into batches
    batches = np.array_split(np.asarray(geoms), n_workers)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            lambda batch: np.array([_make_valid_single(g) for g in batch]),
            batches
        ))
    
    return np.concatenate(results)