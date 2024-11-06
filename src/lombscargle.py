import numpy

from astropy.timeseries import LombScargle

from self_lensing import default_times

maximum_frequency = 0.5
minimum_frequency = 1.0 / max(default_times)


def get_top_period(rv, times=default_times):
    frequency, power = LombScargle(times, rv).autopower(
        maximum_frequency=maximum_frequency, minimum_frequency=minimum_frequency
    )

    period = 1 / frequency
    sort_idx = numpy.argsort(power)
    power = numpy.take(power, sort_idx)
    period = numpy.take(period, sort_idx)
    return period[numpy.argmax(power)]
