# forecast_worker.py
import sys
import multiprocessing as mp
import os
import tempfile
import pickle
import argparse
import traceback
from pathlib import Path
import pandas as pd
# Add project root to sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.forecast import EpicBayesForecaster

# ==========================================================
# CRITICAL FIX: Multiprocessing Start Method (MUST BE FIRST)
# ==========================================================
try:
    if sys.platform != "win32" and mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method("spawn", force=True)
except Exception:
    pass

# Limit threading inside the worker (before numerical libs are imported)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--advi-iter", type=int, required=True)
    parser.add_argument("--sigma-prior", type=float, required=True)
    parser.add_argument("--is-historical", action='store_true')
    parser.add_argument("--learn-bias-variance", action="store_true", help="Enable hierarchical bias & sigma scaling in the model")
    
    args = parser.parse_args()

    try:
        with open(args.input_file, 'rb') as f:
            full_df = pickle.load(f)

        fit_df = full_df.iloc[:-args.steps] if args.is_historical else full_df

        fc = EpicBayesForecaster(fit_df)
        fc.fit(p=args.p, draws=args.draws, method=args.method,
               tune=max(100, args.draws // 2), chains=args.chains,
               cores=args.cores, random_seed=42,
               advi_iter=args.advi_iter, sigma_prior_std=args.sigma_prior,
               learn_bias_variance=args.learn_bias_variance)
        
        forecast_data = fc.forecast(steps=args.steps, draws=args.draws)

        results = {
            "status": "success",
            "forecast_df": forecast_data,
            "full_df": full_df,
            "learn_bias_variance": bool(args.learn_bias_variance)
        }
    except Exception as e:
        results = {
            "status": "error",
            "traceback": f"{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        }
        
    try:
        with open(args.output_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

if __name__ == "__main__":
    main()
