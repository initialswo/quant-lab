import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from qre.data import clean_prices, download_prices
from qre.experiments import run_backtest, sweep


AUTO_ADJUST = True


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None) -> dict:
    default_path = Path("config.yaml")
    with default_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    return config


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _git_dirty() -> bool | None:
    try:
        proc = subprocess.run(["git", "diff", "--quiet"], check=False)
    except Exception:
        return None
    if proc.returncode == 0:
        return False
    if proc.returncode == 1:
        return True
    return None


def build_metadata(
    *,
    timestamp_utc: str,
    mode: str,
    preset: str | None,
    universe: list[str],
    data_period: str,
    data_interval: str,
    auto_adjust: bool,
) -> dict:
    return {
        "timestamp_utc": timestamp_utc,
        "mode": mode,
        "preset": preset,
        "universe": universe,
        "data_period": data_period,
        "data_interval": data_interval,
        "auto_adjust": auto_adjust,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }


def _series_sha256(series: pd.Series) -> str:
    return hashlib.sha256(series.to_numpy().tobytes()).hexdigest()


def run_determinism_self_test(prices: pd.DataFrame, params: dict) -> None:
    out1 = run_backtest(prices, params=params, return_series=True)
    out2 = run_backtest(prices, params=params, return_series=True)

    if not out1.get("ok") or not out2.get("ok"):
        print("Determinism self-test failed: one or both runs returned ok=False")
        raise SystemExit(1)

    for key in ["sharpe", "ann_ret", "ann_vol", "max_dd"]:
        if out1.get(key) != out2.get(key):
            print(f"Determinism self-test failed for metric '{key}'")
            print(f"run1[{key}]={out1.get(key)}")
            print(f"run2[{key}]={out2.get(key)}")
            raise SystemExit(1)

    if "returns" in out1 and "returns" in out2:
        hash1 = _series_sha256(out1["returns"])
        hash2 = _series_sha256(out2["returns"])
        if hash1 != hash2:
            print("Determinism self-test failed: return series hash mismatch")
            print(f"run1 hash={hash1}")
            print(f"run2 hash={hash2}")
            raise SystemExit(1)


def save_single(out: dict, params: dict, outdir: Path, metadata: dict) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = metadata["timestamp_utc"]
    out_path = outdir / f"run_{ts}.json"
    payload = {
        "metadata": metadata,
        "params": params,
        "result": out,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path


def print_top_tables(
    res: pd.DataFrame,
    top_n: int,
    dd_penalty_lambda: float,
    sharpe_floor: float,
    total: int,
) -> None:
    ranked = res.sort_values(["sharpe", "max_dd"], ascending=[False, False])
    ranked_score = res.sort_values(["score", "sharpe"], ascending=[False, False])

    display_cols = [
        "sharpe",
        "ann_ret",
        "ann_vol",
        "max_dd",
        "target_vol",
        "roll_vol",
        "ma_trend",
        "lookback_mom",
        "use_cov",
        "top_k",
    ]

    print(f"\nSweep complete: {len(res)} valid runs (out of {total})")
    print(f"Top {top_n} by Sharpe:\n")

    print(f"\nTop {top_n} by Composite Score (Sharpe - {dd_penalty_lambda}*|MaxDD|):\n")
    print(ranked_score[["score"] + display_cols].head(top_n).to_string(index=False))

    print(ranked[display_cols].head(top_n).to_string(index=False))

    filtered = res[res["sharpe"] >= sharpe_floor].copy()
    if len(filtered) > 0:
        best_dd = filtered.sort_values(["max_dd", "sharpe"], ascending=[False, False]).head(top_n)
        print(f"\nTop {top_n} by MaxDD (Sharpe >= {sharpe_floor:.2f}):\n")
        print(best_dd[display_cols].to_string(index=False))


def run_single(
    prices: pd.DataFrame,
    config: dict,
    preset: str,
    outdir: Path,
    metadata: dict,
    self_test: bool = False,
    print_holdings: bool = False,
    log_holdings: bool = False,
    top_k_override: int | None = None,
) -> Path:
    preset_cfg = config["presets"][preset]
    constants = config["constants"]

    params = {
        **preset_cfg,
        "leverage_hard_cap": constants["leverage_hard_cap"],
        "cash_vol_floor": constants["cash_vol_floor"],
        "bear_target_vol": constants.get("bear_target_vol", 0.10),
        "cost_bps": constants.get("cost_bps", 0.0),
        "smoothing_alpha": constants.get("smoothing_alpha", 1.0),
        "weight_floor": constants.get("weight_floor", 1e-4),
        "top_k": int(preset_cfg.get("top_k", 1)),
        "cash_ticker": str(constants.get("cash_ticker", "SGOV")),
        "abs_mom_threshold": float(constants.get("abs_mom_threshold", 0.0)),
        "weight_method": str(preset_cfg.get("weight_method", constants.get("weight_method", "inv_cov"))),
    }
    if top_k_override is not None:
        params["top_k"] = int(top_k_override)

    out = run_backtest(
        prices,
        params=params,
        return_series=False,
        debug_selection=print_holdings,
        log_holdings=log_holdings,
    )
    if not out.get("ok"):
        raise SystemExit("Preset run failed (not enough data / invalid output).")

    print("\n=== Preset Result ===")
    print(out)
    if print_holdings:
        snaps = out.get("rebalance_holdings", out.get("rebalance_holdings_preview", []))
        print("\n=== Rebalance Holdings (capped snapshots) ===")
        for record in snaps:
            print(record)
    out.pop("rebalance_holdings_preview", None)
    if self_test:
        run_determinism_self_test(prices, params)
    out_path = save_single(out=out, params=params, outdir=outdir, metadata=metadata)
    print(f"\nSaved single run to: {out_path}")
    return out_path


def run_sweep(prices: pd.DataFrame, config: dict, outdir: Path, metadata: dict) -> Path:
    constants = config["constants"]
    grid = {
        **config["sweep"],
        "leverage_hard_cap": constants["leverage_hard_cap"],
        "cash_vol_floor": constants["cash_vol_floor"],
        "bear_target_vol": constants.get("bear_target_vol", 0.10),
        "cost_bps": constants.get("cost_bps", 0.0),
        "smoothing_alpha": constants.get("smoothing_alpha", 1.0),
        "weight_floor": constants.get("weight_floor", 1e-4),
        "top_ks": config["sweep"].get("top_ks", [1]),
        "cash_ticker": str(constants.get("cash_ticker", "SGOV")),
        "abs_mom_threshold": float(constants.get("abs_mom_threshold", 0.0)),
        "weight_method": str(constants.get("weight_method", "inv_cov")),
    }

    dd_penalty_lambda = config["dd_penalty_lambda"]
    res = sweep(prices, grid=grid, dd_penalty_lambda=dd_penalty_lambda)
    if res.empty:
        raise SystemExit("No valid sweep results produced (check data / parameters).")
    total = (
        len(config["sweep"]["target_vols"])
        * len(config["sweep"]["ma_trends"])
        * len(config["sweep"]["roll_vols"])
        * len(config["sweep"]["lookback_moms"])
        * len(config["sweep"]["use_cov"])
        * len(config["sweep"].get("top_ks", [1]))
    )

    print_top_tables(
        res,
        top_n=constants["top_n_results"],
        dd_penalty_lambda=dd_penalty_lambda,
        sharpe_floor=constants["sharpe_floor"],
        total=total,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    csv_metadata = dict(metadata)
    csv_metadata["universe"] = ",".join(metadata["universe"])
    for key, value in csv_metadata.items():
        res[key] = value

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"sweep_{ts}.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved sweep results to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="QRE CLI")
    parser.add_argument("--mode", choices=["single", "sweep"], required=True)
    parser.add_argument("--preset", choices=["performance", "balanced", "defensive"], default="balanced")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="results/")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-holdings", action="store_true")
    parser.add_argument("--log-holdings", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    yf.set_tz_cache_location("/tmp/yf-cache")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    prices = download_prices(
        tickers=config["universe"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        auto_adjust=AUTO_ADJUST,
    )
    prices = clean_prices(prices, cash_ticker=config["constants"].get("cash_ticker"))

    if prices.empty:
        raise SystemExit("No data returned from yfinance.")

    if args.mode == "single":
        metadata = build_metadata(
            timestamp_utc=timestamp_utc,
            mode="single",
            preset=args.preset,
            universe=config["universe"],
            data_period=config["data"]["period"],
            data_interval=config["data"]["interval"],
            auto_adjust=AUTO_ADJUST,
        )
        run_single(
            prices=prices,
            config=config,
            preset=args.preset,
            outdir=outdir,
            metadata=metadata,
            self_test=args.self_test,
            print_holdings=args.print_holdings,
            log_holdings=args.log_holdings,
            top_k_override=args.top_k,
        )
    else:
        metadata = build_metadata(
            timestamp_utc=timestamp_utc,
            mode="sweep",
            preset=None,
            universe=config["universe"],
            data_period=config["data"]["period"],
            data_interval=config["data"]["interval"],
            auto_adjust=AUTO_ADJUST,
        )
        run_sweep(prices=prices, config=config, outdir=outdir, metadata=metadata)


if __name__ == "__main__":
    main()
