import pandas

from tqdm import tqdm

from realistic import load_realistic_timeseries, LOMBSCARGLE_TABLE_PATH
from lombscargle import get_top_period


if __name__ == "__main__":
    results = []
    rts = load_realistic_timeseries()
    for i, ts in tqdm(rts.iterrows(), total=len(rts)):
        top_period = get_top_period(ts["rv"], times=ts["times"])
        results.append(
            {
                "realistic_idx": ts.name,
                "pristine_idx": ts["pristine idx"],
                "top_period": top_period,
                "Porb": ts["Porb"],
            }
        )
    results = pandas.DataFrame(results)
    results.to_parquet(LOMBSCARGLE_TABLE_PATH)
