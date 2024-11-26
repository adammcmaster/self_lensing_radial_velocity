import emcee
import numpy

from astropy import constants, units
from astropy.time import Time

from scipy.optimize import minimize


DEFAULTS = {
    # We just fix impact param as it doesn't make enough difference to change it
    "b": 0 * units.dimensionless_unscaled,
    # We assume a Sun-like star. This shouldn't really matter for RV calculations
    "Mcomp": 1 * units.M_sun,
    "Rcomp": 1 * units.R_sun,
    "Teff": 5500 * units.K,
    # Pick an arbitrary start and end time
    "start_time": Time("2025-01-01"),
    "end_time": Time("2026-01-01"),
}

default_times = list(range(int((DEFAULTS["end_time"] - DEFAULTS["start_time"]).value)))


def gen_sls(a, e, periapsis_phase, macc):
    return SelfLensingSystem(
        a=a * units.au,
        b=DEFAULTS["b"],
        Macc=macc * units.M_sun,
        e=e * units.dimensionless_unscaled,
        periapsis_phase=periapsis_phase * units.rad,
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
    )


def gauss_lnprior(x, mu, sigma=1):
    # https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    gauss_norm = numpy.log(1.0 / (numpy.sqrt(2 * numpy.pi) * sigma))
    return gauss_norm - 0.5 * (x - mu) ** 2 / sigma**2


def rv_chi2(
    params,
    Mcomp,
    Rcomp,
    peak_time,
    times,
    observed_rv,
    observed_rv_err,
    expected_period,
    minimum_period,
    maximum_period,
):
    a, Macc, e, periapsis_phase = params
    cls = SelfLensingSystem

    if a < cls.fit_limits["a"][0] or a > cls.fit_limits["a"][1]:
        return -numpy.inf

    # if b < cls.fit_limits["b"][0] or b > cls.fit_limits["b"][1]:
    #     return -numpy.inf

    if Macc < cls.fit_limits["Macc"][0] or Macc > cls.fit_limits["Macc"][1]:
        return -numpy.inf

    if (
        periapsis_phase < cls.fit_limits["periapsis_phase"][0]
        or periapsis_phase > cls.fit_limits["periapsis_phase"][1]
    ):
        return -numpy.inf

    # e = 1 is parabolic
    if e < cls.fit_limits["e"][0] or e >= cls.fit_limits["e"][1]:
        return -numpy.inf

    model_sls = cls(
        a=a * cls.param_units["a"],
        b=0 * cls.param_units["b"],
        Macc=Macc * cls.param_units["Macc"],
        Mcomp=Mcomp,
        Rcomp=Rcomp,
        e=e * cls.param_units["e"],
        periapsis_phase=periapsis_phase * cls.param_units["periapsis_phase"],
    )

    model_lensing_event = LensingEvent(model_sls, peak_time)
    model_rv = model_lensing_event.rv(times, noise=None)

    rv_chi2 = -numpy.sum(
        (observed_rv.value - model_rv.value) ** 2
        / observed_rv_err.to(units.m / units.s).value ** 2
    ) / len(observed_rv)

    period_err = 0
    if expected_period is not None:
        period_err = gauss_lnprior(model_sls.porb.value, expected_period)

    if minimum_period is not None and model_sls.porb.value < minimum_period:
        return -numpy.inf
    if maximum_period is not None and model_sls.porb.value > maximum_period:
        return -numpy.inf

    return rv_chi2 + period_err


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

    fit_init = {
        "a": 5.0,
        "b": 0.0,
        "Macc": 10,
        "Rcomp": 1,
        "Mcomp": 1,
        "Teff": 5000,
        "e": 0,
        "periapsis_phase": 0,
    }
    fit_limits = {
        "a": (0.01, 50),
        # "a": (1.0, 1.2),
        "b": (0, 10),
        "Macc": (0.1, 50),
        "Rcomp": (0.1, 1000),
        "Mcomp": (0.1, 200),
        "Teff": (100, 100000),
        "e": (0, 1),
        "periapsis_phase": (0, 2 * numpy.pi),
    }

    @classmethod
    @units.quantity_input
    def from_rv_fit(
        cls,
        observed_rv,
        observed_rv_err,
        times,
        Mcomp: units.kg,
        Rcomp: units.m,
        peak_time,
        Teff: units.K = DEFAULT_TEFF,
    ):
        nwalkers = 100
        nsteps = 5000

        p0 = list(
            zip(
                *[
                    numpy.random.uniform(bounds[0], bounds[1], nwalkers)
                    for bounds in (
                        cls.fit_limits["a"],
                        # cls.fit_limits["b"],
                        cls.fit_limits["Macc"],
                        cls.fit_limits["e"],
                        cls.fit_limits["periapsis_phase"],
                    )
                ]
            )
        )

        from lombscargle import get_top_period

        baseline = times[-1].jd - times[0].jd

        expected_period = get_top_period(observed_rv, times=times.jd)
        minimum_period = None
        maximum_period = None
        if expected_period > (0.9 * baseline):
            expected_period = None
            minimum_period = 0.9 * baseline
            maximum_period = 2 * baseline

        from multiprocessing import Pool

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                len(p0[0]),
                rv_chi2,
                args=(
                    Mcomp,
                    Rcomp,
                    peak_time,
                    times,
                    observed_rv,
                    observed_rv_err,
                    expected_period,
                    minimum_period,
                    maximum_period,
                ),
                pool=pool,
                moves=[
                    (emcee.moves.DEMove(), 0.8),
                    (emcee.moves.DESnookerMove(), 0.2),
                ],
            )
            sampler.run_mcmc(
                p0,
                nsteps,
                progress=True,
            )
        flat_samples = sampler.get_chain(flat=True)
        res = flat_samples[numpy.argmax(sampler.get_log_prob(flat=True))]

        out_a, out_Macc, out_e, out_periapsis_phase = res

        return (
            cls(
                a=out_a * cls.param_units["a"],
                b=0 * cls.param_units["b"],
                Macc=out_Macc * cls.param_units["Macc"],
                Rcomp=Rcomp,
                Mcomp=Mcomp,
                Teff=Teff,
                e=out_e * cls.param_units["e"],
                periapsis_phase=out_periapsis_phase
                * cls.param_units["periapsis_phase"],
            ),
            sampler,
        )

    @classmethod
    @units.quantity_input
    def from_rv_fit_scipy(
        cls,
        observed_rv,
        observed_rv_err,
        times,
        Mcomp: units.kg,
        Rcomp: units.m,
        peak_time,
        Teff: units.K = DEFAULT_TEFF,
    ):
        def fit(params):
            a, b, Macc = params

            model_sls = cls(
                a=a * units.AU,
                b=b * units.dimensionless_unscaled,
                Macc=Macc * units.M_sun,
                Mcomp=Mcomp,
                Rcomp=Rcomp,
            )

            model_lensing_event = LensingEvent(model_sls, peak_time)
            model_rv = model_lensing_event.rv(times, noise=None)

            return numpy.sum(
                (observed_rv.value - model_rv.value) ** 2
                / observed_rv_err.to(units.m / units.s).value ** 2
            ) / len(observed_rv)

        res = minimize(
            fit,
            (cls.fit_init["a"], cls.fit_init["b"], cls.fit_init["Macc"]),
            bounds=(cls.fit_limits["a"], cls.fit_limits["b"], cls.fit_limits["Macc"]),
            # method="Nelder-Mead",
        )
        out_a, out_b, out_Macc = res.x

        return (
            cls(
                a=out_a * cls.param_units["a"],
                b=out_b * cls.param_units["b"],
                Macc=out_Macc * cls.param_units["Macc"],
                Rcomp=Rcomp,
                Mcomp=Mcomp,
                Teff=Teff,
            ),
            res,
        )

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
