import pandas

from tqdm import tqdm

from pristine import load_pristine_timeseries, LOMBSCARGLE_TABLE_PATH
from lombscargle import get_top_period


if __name__ == "__main__":
    results = []
    pts = load_pristine_timeseries()
    for i, ts in tqdm(pts.iterrows(), total=len(pts)):
        top_period = get_top_period(ts["rv"])
        results.append(
            {
                "pristine_idx": ts.name,
                "top_period": top_period,
            }
        )
    results = pandas.DataFrame(results)
    results.to_parquet(LOMBSCARGLE_TABLE_PATH)
