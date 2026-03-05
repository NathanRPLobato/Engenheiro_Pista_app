from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

DEFAULT_DB = Path("data/refined/telemetry.sqlite")


@dataclass
class Cfg:
    sqlite_path: str = str(DEFAULT_DB)
    telemetry_table: str = "telemetry_flat"
    laps_table: str = "laps"
    baselines_table: str = "baselines_track_compound"
    out_table: str = "lap_degradation_dataset"
    drop_and_rebuild: bool = False


def _connect(db_path: str):
    return sqlite3.connect(db_path)


def _drop_table(con, name: str):
    con.execute(f"DROP TABLE IF EXISTS {name};")


def build_lap_degradation_dataset(cfg: Cfg):
    con = _connect(cfg.sqlite_path)

    if cfg.drop_and_rebuild:
        _drop_table(con, cfg.out_table)

    print("\n=== BUILD LAP DEGRADATION DATASET (PER LAP) ===")
    print(f"db={cfg.sqlite_path}")
    print(f"out_table={cfg.out_table}")

    # -------------------------------------------------
    # SQL: agrega telemetry por volta
    # -------------------------------------------------
    sql = f"""
    CREATE TABLE {cfg.out_table} AS
    WITH lap_agg AS (
        SELECT
            t.race_id,
            t.trackId,
            t.lap_number,
            l.compound_id,

            MIN(t.lap_time) AS lap_time_start,
            MAX(t.lap_time) AS lap_time_end,
            (MAX(t.lap_time) - MIN(t.lap_time)) AS lap_time,

            AVG(t.throttle) AS throttle_mean,
            AVG(t.brake) AS brake_mean,
            AVG(ABS(t.steering)) AS steering_abs_mean,

            AVG(ABS(t.gforce_Y)) AS gforce_lat_mean,
            AVG(ABS(t.gforce_X)) AS gforce_long_mean,

            AVG(
                (ABS(t.wheel_slip_ratio_0) +
                 ABS(t.wheel_slip_ratio_1) +
                 ABS(t.wheel_slip_ratio_2) +
                 ABS(t.wheel_slip_ratio_3)) / 4.0
            ) AS slip_abs_mean,

            AVG(t.tyre_temp_0 + t.tyre_temp_1 + t.tyre_temp_2 + t.tyre_temp_3) / 4.0
                AS tyre_temp_mean,

            AVG(t.tyre_press_0 + t.tyre_press_1 + t.tyre_press_2 + t.tyre_press_3) / 4.0
                AS tyre_press_mean,

            MIN(t.fuel) AS fuel_start,
            MAX(t.fuel) AS fuel_end

        FROM {cfg.telemetry_table} t
        JOIN {cfg.laps_table} l
          ON l.race_id = t.race_id
         AND l.lap_number = t.lap_number
        WHERE t.lap_time IS NOT NULL
        GROUP BY t.race_id, t.trackId, t.lap_number, l.compound_id
    )

    SELECT
        a.*,

        (a.fuel_start - a.fuel_end) AS fuel_burn,

        b.baseline_mean,
        b.baseline_std,

        (a.lap_time - b.baseline_mean) AS delta_pace,

        -- targets principais
        (a.slip_abs_mean * a.tyre_temp_mean) AS tyre_stress_proxy,
        (a.lap_time - b.baseline_mean) AS degradation_proxy

    FROM lap_agg a
    LEFT JOIN {cfg.baselines_table} b
      ON b.trackId = a.trackId
     AND b.compound_id = a.compound_id
    WHERE a.lap_time > 0
    ;
    """

    try:
        con.executescript(sql)
    except Exception as e:
        con.close()
        raise RuntimeError(f"Erro ao criar dataset: {e}")

    con.close()
    print(f"[ok] tabela criada: {cfg.out_table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite_path", default=str(DEFAULT_DB))
    ap.add_argument("--telemetry_table", default="telemetry_flat")
    ap.add_argument("--laps_table", default="laps")
    ap.add_argument("--baselines_table", default="baselines_track_compound")
    ap.add_argument("--out_table", default="lap_degradation_dataset")
    ap.add_argument("--drop_and_rebuild", action="store_true")
    args = ap.parse_args()

    cfg = Cfg(
        sqlite_path=args.sqlite_path,
        telemetry_table=args.telemetry_table,
        laps_table=args.laps_table,
        baselines_table=args.baselines_table,
        out_table=args.out_table,
        drop_and_rebuild=args.drop_and_rebuild,
    )

    build_lap_degradation_dataset(cfg)


if __name__ == "__main__":
    main()