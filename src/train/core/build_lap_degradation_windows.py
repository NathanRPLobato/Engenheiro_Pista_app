from __future__ import annotations 

import argparse 
import json 
import sqlite3 
from dataclasses import dataclass 
from pathlib import Path 
from typing import List ,Dict ,Optional 

import pandas as pd 


@dataclass 
class Cfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"

    telemetry_table :str ="telemetry_flat"
    laps_table :str ="laps"
    baselines_table :str ="baselines_track_compound"

    out_table :str ="lap_degradation_windows"
    feature_dir :str ="data/feature_store/degradation_windows"

    window_m :float =200.0 
    stride_m :float =100.0 
    min_samples_per_window :int =40 
    max_windows_per_lap :int =60 # igual teu TCN espera

    drop_and_rebuild :bool =False 


    # ---------- utils ----------
def _ensure_dir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _connect (db_path :str )->sqlite3 .Connection :
    con =sqlite3 .connect (db_path )

    # pragmas focados em reduzir I/O e evitar SSD 100% eternamente
    # (WAL ajuda MUITO; temp_store em MEMORY evita muita escrita)
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    con .execute ("PRAGMA temp_store=MEMORY;")
    # cache_size negativo = KB; -200000 200MB
    con .execute ("PRAGMA cache_size=-200000;")
    return con 


def _drop_view_or_table (con :sqlite3 .Connection ,name :str )->None :
    row =con .execute (
    "SELECT type FROM sqlite_master WHERE name=? AND type IN ('table','view');",
    (name ,),
    ).fetchone ()
    if not row :
        return 
    if row [0 ]=="view":
        con .execute (f"DROP VIEW IF EXISTS {name};")
    else :
        con .execute (f"DROP TABLE IF EXISTS {name};")


def _table_cols (con :sqlite3 .Connection ,table :str )->List [str ]:
    df =pd .read_sql (f"PRAGMA table_info({table});",con )
    return df ["name"].tolist ()


def _require_cols (con :sqlite3 .Connection ,table :str ,cols :List [str ])->None :
    have =set (_table_cols (con ,table ))
    missing =[c for c in cols if c not in have ]
    if missing :
        raise SystemExit (f"Tabela {table} sem colunas obrigatórias: {missing}")


def _write_feature_store (con :sqlite3 .Connection ,table :str ,feature_dir :Path )->None :
    cols =_table_cols (con ,table )

    # ids
    id_cols =[
    "race_id","trackId","lap_number","compound_id",
    "win_start_m","win_end_m",
    "window_m","stride_m",
    "n_samples",
    ]
    # targets
    y_cols =["y_pace_window","y_stress_window"]

    ignore =set (id_cols +y_cols )
    x_cols =[c for c in cols if c not in ignore ]

    payload ={
    "table":table ,
    "id_cols":id_cols ,
    "x_cols":x_cols ,
    "y_cols":y_cols ,
    "notes":{
    "pace_target":"y_pace_window = dt_window - baseline_window_time (positivo = pior que baseline)",
    "stress_target":"y_stress_window = 2*slip + brake + 0.5*abs(steer_mean)",
    "speed":"speed2__mean = mean(vx^2+vy^2+vz^2), sqrt se quiser depois",
    },
    }

    _ensure_dir (feature_dir )
    (feature_dir /"features.json").write_text (json .dumps (payload ,indent =2 ),encoding ="utf-8")

    schema =pd .DataFrame (
    con .execute (f"PRAGMA table_info({table});").fetchall (),
    columns =["cid","name","type","notnull","dflt_value","pk"],
    )
    (feature_dir /"schema.json").write_text (schema .to_json (orient ="records",indent =2 ),encoding ="utf-8")


    # ---------- main build ----------
def build_lap_degradation_windows (cfg :Cfg )->None :
    con =_connect (cfg .sqlite_path )

    # validações mínimas
    _require_cols (con ,cfg .telemetry_table ,[
    "race_id","trackId","lap_number","lap_time","lap_distance",
    "throttle","brake","steering","fuel",
    "gforce_X","gforce_Y","angular_vel_Z",
    "tyre_temp_0","tyre_temp_1","tyre_temp_2","tyre_temp_3",
    "tyre_press_0","tyre_press_1","tyre_press_2","tyre_press_3",
    "wheel_slip_ratio_0","wheel_slip_ratio_1","wheel_slip_ratio_2","wheel_slip_ratio_3",
    "velocity_X","velocity_Y","velocity_Z",
    ])
    _require_cols (con ,cfg .laps_table ,["race_id","trackId","lap_number","lap_time","compound_id"])
    _require_cols (con ,cfg .baselines_table ,["trackId","compound_id","baseline_mean"])

    # índices
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.telemetry_table}_rtl ON {cfg.telemetry_table}(race_id, trackId, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.telemetry_table}_rtd ON {cfg.telemetry_table}(race_id, trackId, lap_number, lap_distance);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.laps_table}_rtl ON {cfg.laps_table}(race_id, trackId, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.baselines_table}_tc ON {cfg.baselines_table}(trackId, compound_id);")

    if cfg .drop_and_rebuild :
        _drop_view_or_table (con ,cfg .out_table )

        # recria sempre (pra evitar tabela meia-boca)
    _drop_view_or_table (con ,cfg .out_table )

    # montamos um fallback 'local' (sem tocar no baseline oficial):
    # se baseline_mean do track+compound vier NULL
    # OBS: isso só acontece se teu baseline filtrou demais ou faltou composto na pista.
    sql =f"""
    CREATE TABLE {cfg.out_table} AS
    WITH
    tel AS (
      SELECT
        CAST(t.race_id AS TEXT) AS race_id,
        CAST(t.trackId AS TEXT) AS trackId,
        CAST(t.lap_number AS INT) AS lap_number,
        CAST(t.lap_distance AS REAL) AS lap_distance,
        CAST(t.lap_time AS REAL) AS lap_time,

        CAST(t.throttle AS REAL) AS throttle,
        CAST(t.brake AS REAL) AS brake,
        CAST(t.steering AS REAL) AS steering,
        CAST(t.fuel AS REAL) AS fuel,

        CAST(t.gforce_X AS REAL) AS gforce_X,
        CAST(t.gforce_Y AS REAL) AS gforce_Y,
        CAST(t.angular_vel_Z AS REAL) AS angular_vel_Z,

        CAST(t.tyre_temp_0 AS REAL) AS tyre_temp_0,
        CAST(t.tyre_temp_1 AS REAL) AS tyre_temp_1,
        CAST(t.tyre_temp_2 AS REAL) AS tyre_temp_2,
        CAST(t.tyre_temp_3 AS REAL) AS tyre_temp_3,

        CAST(t.tyre_press_0 AS REAL) AS tyre_press_0,
        CAST(t.tyre_press_1 AS REAL) AS tyre_press_1,
        CAST(t.tyre_press_2 AS REAL) AS tyre_press_2,
        CAST(t.tyre_press_3 AS REAL) AS tyre_press_3,

        CAST(t.wheel_slip_ratio_0 AS REAL) AS wheel_slip_ratio_0,
        CAST(t.wheel_slip_ratio_1 AS REAL) AS wheel_slip_ratio_1,
        CAST(t.wheel_slip_ratio_2 AS REAL) AS wheel_slip_ratio_2,
        CAST(t.wheel_slip_ratio_3 AS REAL) AS wheel_slip_ratio_3,

        CAST(t.velocity_X AS REAL) AS velocity_X,
        CAST(t.velocity_Y AS REAL) AS velocity_Y,
        CAST(t.velocity_Z AS REAL) AS velocity_Z
      FROM {cfg.telemetry_table} t
      WHERE t.lap_number >= 0
        AND t.lap_time IS NOT NULL AND t.lap_time > 0
        AND t.lap_distance IS NOT NULL
    ),
    laps_clean AS (
      SELECT
        CAST(race_id AS TEXT) AS race_id,
        CAST(trackId AS TEXT) AS trackId,
        CAST(lap_number AS INT) AS lap_number,
        CAST(lap_time AS REAL) AS lap_time_laps,
        CAST(compound_id AS INT) AS compound_id
      FROM {cfg.laps_table}
      WHERE lap_number >= 0 AND lap_time IS NOT NULL AND lap_time > 0
    ),
    joined AS (
      SELECT
        te.*,
        l.compound_id,
        l.lap_time_laps
      FROM tel te
      JOIN laps_clean l
        ON l.race_id = te.race_id
       AND l.trackId = te.trackId
       AND l.lap_number = te.lap_number
    ),
    limits AS (
      SELECT
        race_id, trackId, lap_number, compound_id,
        MAX(lap_distance) AS lap_dist_max,
        MAX(lap_time_laps) AS lap_time_total
      FROM joined
      GROUP BY race_id, trackId, lap_number, compound_id
    ),
    ks AS (
      SELECT 0 AS k UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
      UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
      UNION ALL SELECT 10 UNION ALL SELECT 11 UNION ALL SELECT 12 UNION ALL SELECT 13 UNION ALL SELECT 14
      UNION ALL SELECT 15 UNION ALL SELECT 16 UNION ALL SELECT 17 UNION ALL SELECT 18 UNION ALL SELECT 19
      UNION ALL SELECT 20 UNION ALL SELECT 21 UNION ALL SELECT 22 UNION ALL SELECT 23 UNION ALL SELECT 24
      UNION ALL SELECT 25 UNION ALL SELECT 26 UNION ALL SELECT 27 UNION ALL SELECT 28 UNION ALL SELECT 29
      UNION ALL SELECT 30 UNION ALL SELECT 31 UNION ALL SELECT 32 UNION ALL SELECT 33 UNION ALL SELECT 34
      UNION ALL SELECT 35 UNION ALL SELECT 36 UNION ALL SELECT 37 UNION ALL SELECT 38 UNION ALL SELECT 39
      UNION ALL SELECT 40 UNION ALL SELECT 41 UNION ALL SELECT 42 UNION ALL SELECT 43 UNION ALL SELECT 44
      UNION ALL SELECT 45 UNION ALL SELECT 46 UNION ALL SELECT 47 UNION ALL SELECT 48 UNION ALL SELECT 49
      UNION ALL SELECT 50 UNION ALL SELECT 51 UNION ALL SELECT 52 UNION ALL SELECT 53 UNION ALL SELECT 54
      UNION ALL SELECT 55 UNION ALL SELECT 56 UNION ALL SELECT 57 UNION ALL SELECT 58 UNION ALL SELECT 59
    ),
    windows AS (
      SELECT
        lim.race_id, lim.trackId, lim.lap_number, lim.compound_id,
        CAST({float(cfg.window_m)} AS REAL) AS window_m,
        CAST({float(cfg.stride_m)} AS REAL) AS stride_m,
        CAST((k * {float(cfg.stride_m)}) AS REAL) AS win_start_m,
        CAST((k * {float(cfg.stride_m)} + {float(cfg.window_m)}) AS REAL) AS win_end_m
      FROM limits lim
      JOIN ks
      WHERE k < {int(cfg.max_windows_per_lap)}
        AND (k * {float(cfg.stride_m)}) < lim.lap_dist_max
    ),
    wdata AS (
      SELECT
        w.race_id, w.trackId, w.lap_number, w.compound_id,
        w.window_m, w.stride_m, w.win_start_m, w.win_end_m,

        COUNT(*) AS n_samples,

        AVG(j.throttle) AS throttle__mean,
        (AVG(j.throttle*j.throttle) - AVG(j.throttle)*AVG(j.throttle)) AS throttle__var,
        MAX(j.throttle) AS throttle__max,
        MIN(j.throttle) AS throttle__min,

        AVG(j.brake) AS brake__mean,
        (AVG(j.brake*j.brake) - AVG(j.brake)*AVG(j.brake)) AS brake__var,
        MAX(j.brake) AS brake__max,
        MIN(j.brake) AS brake__min,

        AVG(j.steering) AS steering__mean,
        (AVG(j.steering*j.steering) - AVG(j.steering)*AVG(j.steering)) AS steering__var,
        MAX(j.steering) AS steering__max,
        MIN(j.steering) AS steering__min,

        AVG(j.fuel) AS fuel__mean,

        AVG(j.gforce_X) AS gforce_X__mean,
        AVG(j.gforce_Y) AS gforce_Y__mean,
        AVG(j.angular_vel_Z) AS angular_vel_Z__mean,

        AVG((ABS(j.wheel_slip_ratio_0)+ABS(j.wheel_slip_ratio_1)+ABS(j.wheel_slip_ratio_2)+ABS(j.wheel_slip_ratio_3))/4.0) AS slip_abs__mean,
        MAX((ABS(j.wheel_slip_ratio_0)+ABS(j.wheel_slip_ratio_1)+ABS(j.wheel_slip_ratio_2)+ABS(j.wheel_slip_ratio_3))/4.0) AS slip_abs__max,

        AVG((j.tyre_temp_0+j.tyre_temp_1+j.tyre_temp_2+j.tyre_temp_3)/4.0) AS tyre_temp_mean,
        AVG((j.tyre_press_0+j.tyre_press_1+j.tyre_press_2+j.tyre_press_3)/4.0) AS tyre_press_mean,

        AVG(j.velocity_X*j.velocity_X + j.velocity_Y*j.velocity_Y + j.velocity_Z*j.velocity_Z) AS speed2__mean,

        -- dt_window: aproximação pelo tempo de volta * fração da distância
        (SELECT lim2.lap_time_total * (w.window_m / NULLIF(lim2.lap_dist_max, 0.0))
         FROM limits lim2
         WHERE lim2.race_id=w.race_id AND lim2.trackId=w.trackId AND lim2.lap_number=w.lap_number AND lim2.compound_id=w.compound_id
         LIMIT 1
        ) AS dt_window,

        (SELECT lim3.lap_dist_max
         FROM limits lim3
         WHERE lim3.race_id=w.race_id AND lim3.trackId=w.trackId AND lim3.lap_number=w.lap_number AND lim3.compound_id=w.compound_id
         LIMIT 1
        ) AS lap_dist_max
      FROM windows w
      JOIN joined j
        ON j.race_id=w.race_id AND j.trackId=w.trackId AND j.lap_number=w.lap_number AND j.compound_id=w.compound_id
       AND j.lap_distance >= w.win_start_m AND j.lap_distance < w.win_end_m
      GROUP BY
        w.race_id, w.trackId, w.lap_number, w.compound_id,
        w.window_m, w.stride_m, w.win_start_m, w.win_end_m
    ),
    baseline_track AS (
      SELECT trackId, AVG(baseline_mean) AS baseline_track_mean
      FROM {cfg.baselines_table}
      GROUP BY trackId
    )
    SELECT
      wd.*,

      -- baseline oficial (track+compound)
      b.baseline_mean AS baseline_mean_tc,

      -- fallback local (track-only) se faltou tc
      bt.baseline_track_mean AS baseline_mean_track,

      COALESCE(b.baseline_mean, bt.baseline_track_mean) AS baseline_mean,

      -- baseline_window_time = baseline lap time * fração da distância (janela / dist_max da volta)
      (COALESCE(b.baseline_mean, bt.baseline_track_mean) * (wd.window_m / NULLIF(wd.lap_dist_max, 0.0))) AS baseline_window_time,

      -- targets
      (wd.dt_window - (COALESCE(b.baseline_mean, bt.baseline_track_mean) * (wd.window_m / NULLIF(wd.lap_dist_max, 0.0)))) AS y_pace_window,

      (2.0*wd.slip_abs__mean + 1.0*wd.brake__mean + 0.5*ABS(wd.steering__mean)) AS y_stress_window

    FROM wdata wd
    LEFT JOIN {cfg.baselines_table} b
      ON b.trackId = wd.trackId AND b.compound_id = wd.compound_id
    LEFT JOIN baseline_track bt
      ON bt.trackId = wd.trackId
    ;
    """

    con .execute ("BEGIN;")
    try :
        con .execute (sql )
        con .commit ()
    except Exception :
        con .rollback ()
        con .close ()
        raise 

        # filtros finais (garantir dataset limpo)
    con .execute (f"DELETE FROM {cfg.out_table} WHERE n_samples < {int(cfg.min_samples_per_window)};")
    con .execute (f"DELETE FROM {cfg.out_table} WHERE baseline_mean IS NULL;")
    con .commit ()

    # índices finais
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}_rtl ON {cfg.out_table}(race_id, trackId, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}_tc ON {cfg.out_table}(trackId, compound_id);")
    con .commit ()

    # feature_store
    _write_feature_store (con ,cfg .out_table ,Path (cfg .feature_dir ))

    # sanity prints
    n =con .execute (f"SELECT COUNT(*) FROM {cfg.out_table};").fetchone ()[0 ]
    n_null =con .execute (f"SELECT COUNT(*) FROM {cfg.out_table} WHERE baseline_mean IS NULL;").fetchone ()[0 ]
    n_fb =con .execute (f"SELECT COUNT(*) FROM {cfg.out_table} WHERE baseline_mean_tc IS NULL AND baseline_mean_track IS NOT NULL;").fetchone ()[0 ]
    con .close ()

    print (f"[ok] built {cfg.out_table}: rows={n} baseline_null={n_null} fallback_track_only_rows={n_fb}")
    print (f"[ok] feature_store: {cfg.feature_dir}")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--telemetry",default ="telemetry_flat")
    ap .add_argument ("--laps",default ="laps")
    ap .add_argument ("--baselines",default ="baselines_track_compound")
    ap .add_argument ("--out_table",default ="lap_degradation_windows")
    ap .add_argument ("--feature_dir",default ="data/feature_store/degradation_windows")
    ap .add_argument ("--window_m",type =float ,default =200.0 )
    ap .add_argument ("--stride_m",type =float ,default =100.0 )
    ap .add_argument ("--min_samples_per_window",type =int ,default =40 )
    ap .add_argument ("--max_windows_per_lap",type =int ,default =60 )
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    telemetry_table =args .telemetry ,
    laps_table =args .laps ,
    baselines_table =args .baselines ,
    out_table =args .out_table ,
    feature_dir =args .feature_dir ,
    window_m =float (args .window_m ),
    stride_m =float (args .stride_m ),
    min_samples_per_window =int (args .min_samples_per_window ),
    max_windows_per_lap =int (args .max_windows_per_lap ),
    drop_and_rebuild =bool (args .drop_and_rebuild ),
    )

    print ("\n=== BUILD LAP DEGRADATION WINDOWS (uses official baselines, no overwrite) ===")
    print (f"db={cfg.sqlite_path}")
    print (f"telemetry={cfg.telemetry_table} laps={cfg.laps_table} baselines={cfg.baselines_table}")
    print (f"out_table={cfg.out_table}")
    print (f"feature_dir={cfg.feature_dir}")
    print (f"window_m={cfg.window_m} stride_m={cfg.stride_m} min_samples={cfg.min_samples_per_window} max_windows_per_lap={cfg.max_windows_per_lap}")
    print (f"drop_and_rebuild={cfg.drop_and_rebuild}")

    build_lap_degradation_windows (cfg )


if __name__ =="__main__":
    main ()