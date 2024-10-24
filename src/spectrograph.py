import numpy

import astropy.units as units


class Spectrograph(object):
    NAME = "Generic Spectrograph"
    RV_ACCURACY = 1 * units.m / units.s
    STELLAR_NOISE = 10 * units.m / units.s

    all = []

    @classmethod
    def add_rv_noise(cls, rv):
        # abs_rv = abs(rv.value)
        len_rv = len(rv)
        return (
            rv.value
            # Stellar variations
            + numpy.random.normal(0, cls.STELLAR_NOISE.to(rv.unit).value, len_rv)
            # Spectroscope accuracy
            + numpy.random.normal(0, cls.RV_ACCURACY.to(rv.unit).value, len_rv)
        ) * rv.unit


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
