import logging

import numpy

import astropy.units as units

logger = logging.getLogger(__name__)


class Spectrograph(object):
    NAME = "Generic Spectrograph"
    RV_ACCURACY = 1 * units.m / units.s
    STELLAR_NOISE = 10 * units.m / units.s

    all = []

    @classmethod
    def add_rv_noise(cls, rv, accuracy=None):
        if accuracy is None:
            accuracy = cls.RV_ACCURACY
        len_rv = len(rv)
        logger.debug(
            "Adding RV noise with %s: len=%d stellar_noise=%s accuracy=%s",
            cls.__name__,
            len_rv,
            cls.STELLAR_NOISE,
            accuracy,
        )
        return (
            rv.value
            # Stellar variations
            + numpy.random.normal(0, cls.STELLAR_NOISE.to(rv.unit).value, len_rv)
            # Spectroscope accuracy
            + numpy.random.normal(0, accuracy.to(rv.unit).value, len_rv)
        ) * rv.unit

    @classmethod
    def total_error(cls, unit=units.m / units.s, accuracy=None):
        if accuracy is None:
            accuracy = cls.RV_ACCURACY
        error = numpy.sqrt(
            cls.STELLAR_NOISE.to(unit).value ** 2 + accuracy.to(unit).value ** 2
        )
        logger.debug(
            "Computed total RV error with %s: unit=%s accuracy=%s error=%s",
            cls.__name__,
            unit,
            accuracy,
            error,
        )
        return error


class ARCES(Spectrograph):
    NAME = "APO ARCES"
    RV_ACCURACY = 0.5 * units.km / units.s


Spectrograph.all.append(ARCES)


class ESPRESSO(Spectrograph):
    NAME = "ESO ESPRESSO"
    RV_ACCURACY = 0.097 * units.m / units.s


Spectrograph.all.append(ESPRESSO)


class FourMost(Spectrograph):
    NAME = "4Most"
    RV_ACCURACY = 2 * units.m / units.s


Spectrograph.all.append(FourMost)


class HARPS(Spectrograph):
    NAME = "ESO HARPS"
    RV_ACCURACY = 0.97 * units.m / units.s


Spectrograph.all.append(HARPS)


class HARPSN(Spectrograph):
    NAME = "TNG HARPS-N"
    RV_ACCURACY = 0.3 * units.m / units.s


Spectrograph.all.append(HARPSN)


class HRS(Spectrograph):
    NAME = "SALT HRS"
    RV_ACCURACY = 3 * units.m / units.s


Spectrograph.all.append(HRS)


class NIRPS(Spectrograph):
    NAME = "ESO NIRPS"
    RV_ACCURACY = 1 * units.m / units.s


Spectrograph.all.append(NIRPS)


class RSS(Spectrograph):
    NAME = "SALT RSS"
    RV_ACCURACY = 6 * units.km / units.s


Spectrograph.all.append(RSS)
