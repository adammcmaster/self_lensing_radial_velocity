import numpy

from astropy import constants, units
from astropy.time import Time


class LensingEvent(object):
    DEFAULT_NOISE_GENERATOR = None
    DEFAULT_TIME_STEP = 1 * units.day

    def __init__(
        self,
        lensing_system,
        peak_time,
    ):
        self.lensing_system = lensing_system
        self.peak_time = peak_time

    def rv(self, times, noise=DEFAULT_NOISE_GENERATOR):
        time_offsets = (times - self.peak_time).to(units.day)
        phases = self.lensing_system.time_to_phase(time_offsets)
        rv = self.lensing_system.rv(phases)
        if noise is not None:
            rv = noise.add_rv_noise(rv)
        return rv

    def rv_range(
        self, start, end, step=DEFAULT_TIME_STEP, noise=DEFAULT_NOISE_GENERATOR
    ):
        """
        Generate predicted RV values for the event for the given time range.

        - start: The start time of the range.
        - end: the end time of the range.
        - step: The step size of the range (i.e. time in days between samples). (Default: 1)
        - noise: distribution from which to add noise. Valid value are poisson or None.
        """
        return self.rv(self.time_range(start, end, step=step), noise=noise)

    def time_range(self, start, end, step=DEFAULT_TIME_STEP, repeats=1):
        times = []
        for time in numpy.arange(start, end, step=step):
            for i in range(repeats):
                times.append(time)
        return Time(times)


class SelfLensingSystem(object):
    param_units = {
        "a": units.au,
        "b": units.dimensionless_unscaled,
        "Macc": units.M_sun,
        "Rcomp": units.R_sun,
        "Mcomp": units.M_sun,
        "Teff": units.K,
        "e": units.dimensionless_unscaled,
        "periapsis_phase": units.rad,
    }
    DEFAULT_TEFF = 5000 * units.K
    DEFAULT_PERIAPSIS = 0 * units.rad
    DEFAULT_E = 0 * units.dimensionless_unscaled

    @units.quantity_input
    def __init__(
        self,
        a: units.m,
        b: units.dimensionless_unscaled,
        Macc: units.kg,
        Mcomp: units.kg,
        Rcomp: units.m,
        Teff: units.K = DEFAULT_TEFF,
        e: units.dimensionless_unscaled = DEFAULT_E,
        periapsis_phase: units.rad = DEFAULT_PERIAPSIS,
    ):
        self.b = b
        self.a = a.to(self.param_units["a"])
        self.Macc = Macc.to(self.param_units["Macc"])
        self.Rcomp = Rcomp.to(self.param_units["Rcomp"])
        self.Mcomp = Mcomp.to(self.param_units["Mcomp"])
        self.Teff = Teff.to(self.param_units["Teff"])
        self.e = e
        self.periapsis_phase = periapsis_phase.to(self.param_units["periapsis_phase"])

    @property
    def a_acc(self):
        return self.a * self.Mcomp / self.Mtot

    @property
    def a_comp(self):
        return self.a * self.Macc / self.Mtot

    @property
    def b_max(self):
        """
        Maximum b where SL is possible as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return 1 + self.Rcomp / self.rE

    @property
    def Mtot(self):
        return self.Macc + self.Mcomp

    @property
    @units.quantity_input
    def porb(self) -> units.day:
        return (
            2
            * numpy.pi
            * numpy.sqrt(self.a**3 / (constants.G * self.Mtot)).to(units.day)
        )

    @property
    def rE(self) -> units.R_sun:
        return numpy.sqrt((4 * constants.G * self.Macc * self.a) / constants.c**2).to(
            units.R_sun
        )

    def rv(self, phase):
        """
        Radial velocity. Applies phase angle to orbital velocity.
        """
        return self.v_comp(phase) * numpy.sin(phase) * self.sin_i

    def rv_deriv(self, phase):
        """
        Derivative of radial velocity.
        """
        return self.v_comp(phase) * numpy.cos(phase) * self.sin_i

    @property
    def sin_i(self):
        """
        sin of orbital inclination

        As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
        Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
        Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
        """
        psi = numpy.arcsin(
            (self.b * self.rE) / self.a
        )  # The angular inclination of the source
        return numpy.sin((0.5 * numpy.pi) * units.rad - psi)

    @property
    def tau_eff(self):
        """
        Effective crossing time as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return (
            ((self.porb * (self.rE + self.Rcomp)) / (numpy.pi * self.a * self.sin_i))
            * numpy.sqrt(1 - (self.b / self.b_max) ** 2)
        ).to(units.day)

    @units.quantity_input
    def time_to_phase(self, time: units.day) -> units.rad:
        """
        Convert time offsets to angular orbital phases.
        """
        time = time % self.porb
        return (((time / self.porb).decompose() * 2 * numpy.pi)) * units.rad

    def v(self, a, phase) -> units.m / units.s:
        """
        Orbital velocity.
        """
        r = (a * (1 - self.e**2)) / (
            1 + self.e * numpy.cos(phase - self.periapsis_phase)
        )
        v = numpy.sqrt((constants.G * self.Mtot) * (2 / r - 1 / a))
        return v.to(units.m / units.s)

    def v_tot(self, phase) -> units.m / units.s:
        return self.v(self.a, phase)

    def v_acc(self, phase) -> units.m / units.s:
        return self.v(self.a_acc, phase)

    def v_comp(self, phase) -> units.m / units.s:
        return self.v(self.a_comp, phase)
