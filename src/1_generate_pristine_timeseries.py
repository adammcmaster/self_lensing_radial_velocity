"""
This generates a set of RV timeseries with no errors and a uniform observing cadence.

These timeseries can then form the basis of more realistic ones generated later.
"""

from pristine import generate_pristine_timeseries


if __name__ == "__main__":
    generate_pristine_timeseries()
