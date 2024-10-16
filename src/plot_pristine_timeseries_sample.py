import numpy
import seaborn

from astropy import units

from matplotlib import pyplot

from pathlib import Path

from pristine import PARAMS, DATA_PATH, gen_sls

OUT_PATH = Path(__file__).parent / Path("../plots/pristine")
OUT_PATH.mkdir(exist_ok=True, parents=True)

times = list(range(int((PARAMS["end_time"] - PARAMS["start_time"]).value)))

if __name__ == "__main__":
    for a in PARAMS["a_sample"]:
        with numpy.load(DATA_PATH / f"{a}.npz") as a_timeseries:
            for ts_array in a_timeseries.values():
                idx = numpy.random.randint(ts_array.shape[0], size=10)
                for i in idx:
                    ts = ts_array[i, :]
                    _a, e, periapsis_phase, macc = ts[:4]
                    sls = gen_sls(a, e, periapsis_phase, macc)
                    plot_path = (
                        OUT_PATH
                        / str(a)
                        / str(e)
                        / str(periapsis_phase)
                        / str(sls.Mtot.value)
                    )
                    plot_path.mkdir(exist_ok=True, parents=True)

                    seaborn.scatterplot({"time": times, "rv": ts[4:]}, x="time", y="rv")
                    pyplot.savefig(plot_path / "unfolded.png")
                    pyplot.close()

                    porb = sls.porb.to(units.day).value
                    seaborn.scatterplot(
                        {"time": times % porb, "rv": ts[4:]},
                        x="time",
                        y="rv",
                    )

                    pyplot.savefig(plot_path / "folded.png")
                    pyplot.close()
