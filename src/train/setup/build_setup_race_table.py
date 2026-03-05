# arquivo: src/train/setup/build_setup_race_table.py
from __future__ import annotations 

import argparse 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import List ,Dict ,Tuple 

import numpy as np 
import pandas as pd 


DEFAULT_SQLITE =Path ("data/refined/telemetry.sqlite")
DEFAULT_OUT_TABLE ="setup_race"
DEFAULT_FEATURE_DIR =Path ("data/feature_store/setup_race")


# -----------------------------
# Helpers SQLite
# -----------------------------
def _obj_type (con :sqlite3 .Connection ,name :str )->str |None :
    row =con .execute (
    "SELECT type FROM sqlite_master WHERE name=? AND type IN ('table','view')",
    (name ,),
    ).fetchone ()
    return row [0 ]if row else None 


def _drop_view_or_table (con :sqlite3 .Connection ,name :str )->None :
    t =_obj_type (con ,name )
    if t =="view":
        con .execute (f"DROP VIEW IF EXISTS {name};")
    elif t =="table":
        con .execute (f"DROP TABLE IF EXISTS {name};")


def _table_exists (con :sqlite3 .Connection ,name :str )->bool :
    return _obj_type (con ,name )=="table"


def _get_cols (con :sqlite3 .Connection ,table :str )->List [str ]:
    return [r [1 ]for r in con .execute (f"PRAGMA table_info({table});").fetchall ()]


def _pick_existing (cols :List [str ],requested :List [str ])->List [str ]:
    s =set (cols )
    return [c for c in requested if c in s ]


def _cast_numeric_safe (df :pd .DataFrame ,cols :List [str ])->pd .DataFrame :
    for c in cols :
        if c in df .columns :
            df [c ]=pd .to_numeric (df [c ],errors ="coerce")
    return df 


def _first_valid (s :pd .Series ):
    s2 =s .dropna ()
    return s2 .iloc [0 ]if len (s2 )else np .nan 


def _to_json (path :Path ,obj :dict )->None :
# sem depender do json pra manter compatível com teu estilo de projeto
    path .write_text (pd .Series (obj ).to_json (),encoding ="utf-8")


    # -----------------------------
    # Config
    # -----------------------------
@dataclass 
class SetupRaceCfg :
    sqlite_path :str 
    telemetry_table :str 
    laps_table :str 
    out_table :str 
    feature_dir :str 
    drop_and_rebuild :bool 


    # -----------------------------
    # Builder
    # -----------------------------
def build_setup_race_table (db_path :Path ,cfg :SetupRaceCfg )->pd .DataFrame :
    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")

    # valida tabelas/colunas
    tel_cols =_get_cols (con ,cfg .telemetry_table )
    laps_cols =_get_cols (con ,cfg .laps_table )

    required_tel =["race_id","trackId","lap_number","lap_time"]
    required_laps =["race_id","trackId","lap_number","compound_id"]

    for c in required_tel :
        if c not in tel_cols :
            raise SystemExit (f"[ERRO] coluna obrigatória ausente em {cfg.telemetry_table}: {c}")
    for c in required_laps :
        if c not in laps_cols :
            raise SystemExit (f"[ERRO] coluna obrigatória ausente em {cfg.laps_table}: {c}")

            # setup (fixo por corrida)
    setup_candidates =[
    "wing_setup_0","wing_setup_1",
    "diff_onThrottle_setup","diff_offThrottle_setup",
    "camber_setup_0","camber_setup_1","camber_setup_2","camber_setup_3",
    "toe_setup_0","toe_setup_1","toe_setup_2","toe_setup_3",
    "susp_spring_setup_0","susp_spring_setup_1","susp_spring_setup_2","susp_spring_setup_3",
    "arb_setup_0","arb_setup_1",
    "susp_height_setup_0","susp_height_setup_1","susp_height_setup_2","susp_height_setup_3",
    "brake_press_setup","brake_bias_setup","brake_engine_setup",
    "tyre_press_setup_0","tyre_press_setup_1","tyre_press_setup_2","tyre_press_setup_3",
    "ballast_setup","fuel_setup",
    "traction_ctrl_setup","abs_setup",
    ]
    setup_cols =_pick_existing (tel_cols ,setup_candidates )

    # dinâmica/pilotagem (agregada por corrida)
    dyn_candidates =[
    "throttle","brake","clutch","steering",
    "fuel","rpm","gear",
    "gforce_X","gforce_Y","gforce_Z",
    "angular_vel_X","angular_vel_Y","angular_vel_Z",
    "wheel_slip_ratio_0","wheel_slip_ratio_1","wheel_slip_ratio_2","wheel_slip_ratio_3",
    "wheel_slip_angle_0","wheel_slip_angle_1","wheel_slip_angle_2","wheel_slip_angle_3",
    "tyre_temp_0","tyre_temp_1","tyre_temp_2","tyre_temp_3",
    "tyre_press_0","tyre_press_1","tyre_press_2","tyre_press_3",
    "lap_distance",
    ]
    dyn_cols =_pick_existing (tel_cols ,dyn_candidates )

    print ("\n=== BUILD SETUP RACE DATASET (1 linha por corrida) ===")
    print (f"db={db_path}")
    print (f"telemetry={cfg.telemetry_table}  laps={cfg.laps_table}")
    print (f"setup_cols={len(setup_cols)} dyn_cols={len(dyn_cols)}")

    if cfg .drop_and_rebuild :
        _drop_view_or_table (con ,cfg .out_table )
        con .commit ()

        # 1) telemetry sem compound_id
    sel_cols =["race_id","trackId","lap_number","lap_time"]+setup_cols +dyn_cols 
    sel_cols =list (dict .fromkeys (sel_cols ))# unique mantendo ordem

    sql_tel =f"""
        SELECT {", ".join(sel_cols)}
        FROM {cfg.telemetry_table}
        WHERE lap_number >= 0
          AND lap_time IS NOT NULL AND lap_time > 0
    """
    t0 =time .time ()
    df_tel =pd .read_sql (sql_tel ,con )
    t1 =time .time ()

    if df_tel .empty :
        raise SystemExit ("[ERRO] telemetry não retornou linhas (check filtros).")

        # cast numérico
    numeric_to_cast =["lap_time"]+setup_cols +dyn_cols 
    df_tel =_cast_numeric_safe (df_tel ,numeric_to_cast )

    # 2) compound por corrida via laps (mode ponderado por n_laps)
    sql_comp =f"""
        SELECT race_id, trackId, compound_id, COUNT(*) AS n_laps
        FROM {cfg.laps_table}
        GROUP BY race_id, trackId, compound_id
    """
    df_comp =pd .read_sql (sql_comp ,con )
    if df_comp .empty :
        raise SystemExit ("[ERRO] laps não retornou compound_id (tabela vazia?).")

    df_comp ["n_laps"]=pd .to_numeric (df_comp ["n_laps"],errors ="coerce").fillna (0 ).astype (int )
    df_comp =df_comp .sort_values (["race_id","trackId","n_laps"],ascending =[True ,True ,False ])

    df_mode =df_comp .groupby (["race_id","trackId"],as_index =False ).first ()
    df_mode =df_mode .rename (columns ={"compound_id":"compound_id_mode","n_laps":"compound_mode_n_laps"})

    df_div =df_comp .groupby (["race_id","trackId"],as_index =False ).agg (
    n_compounds =("compound_id","nunique"),
    laps_total =("n_laps","sum"),
    )

    df_comp_race =df_mode .merge (df_div ,on =["race_id","trackId"],how ="left")

    # 3) agrega telemetry por corrida
    g =df_tel .groupby (["race_id","trackId"],as_index =False )

    agg :Dict [str ,Tuple [str ,str ]]={
    "lap_time__mean":("lap_time","mean"),
    "lap_time__std":("lap_time","std"),
    "lap_time__min":("lap_time","min"),
    "lap_time__max":("lap_time","max"),
    "n_samples":("lap_time","size"),
    "n_laps_obs":("lap_number","nunique"),
    }

    for c in dyn_cols :
        if c =="lap_distance":
            agg [f"{c}__max"]=(c ,"max")
        else :
            agg [f"{c}__mean"]=(c ,"mean")
            agg [f"{c}__max"]=(c ,"max")

    df_race =g .agg (**{k :pd .NamedAgg (column =v [0 ],aggfunc =v [1 ])for k ,v in agg .items ()})

    # setup: constante por corrida -> first_valid
    if setup_cols :
    # IMPORTANTE: isso já volta com race_id e trackId, então não inventa insert
        df_setup =g [["race_id","trackId"]+setup_cols ].agg (_first_valid )
        # garante colunas certas (algumas versões podem 'embaralhar' tipos)
        df_setup =df_setup [["race_id","trackId"]+setup_cols ]
        df_race =df_race .merge (df_setup ,on =["race_id","trackId"],how ="left")

        # 4) junta compound stats
    df_race =df_race .merge (df_comp_race ,on =["race_id","trackId"],how ="left")

    # 5) targets auxiliares: burn de corrida (se fuel existe)
    if "fuel"in df_tel .columns :
        df_fuel =df_tel .groupby (["race_id","trackId"],as_index =False ).agg (
        fuel__min =("fuel","min"),
        fuel__max =("fuel","max"),
        )
        df_fuel ["y_burn_race"]=df_fuel ["fuel__max"]-df_fuel ["fuel__min"]
        df_race =df_race .merge (df_fuel ,on =["race_id","trackId"],how ="left")

    df_race =df_race .sort_values (["trackId","race_id"]).reset_index (drop =True )

    # 6) grava SQLite
    if cfg .drop_and_rebuild and _table_exists (con ,cfg .out_table ):
        con .execute (f"DELETE FROM {cfg.out_table};")
        con .commit ()

    df_race .to_sql (cfg .out_table ,con ,if_exists ="replace",index =False )
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}__race ON {cfg.out_table}(race_id);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}__track ON {cfg.out_table}(trackId);")
    con .commit ()
    con .close ()

    print (f"[ok] telemetry read em {t1 - t0:.2f}s  rows={len(df_tel)}")
    print (f"[ok] out_table={cfg.out_table} rows={len(df_race)}")

    # 7) feature store
    outdir =Path (cfg .feature_dir )
    outdir .mkdir (parents =True ,exist_ok =True )

    id_cols =["race_id"]
    cat_cols =["trackId"]

    # numeric = tudo que não é id/cat
    numeric_cols =[c for c in df_race .columns if c not in (id_cols +cat_cols )]

    # remove colunas object que não viraram número (e não são desejáveis)
    drop_obj =[]
    for c in numeric_cols :
        if df_race [c ].dtype =="object":
            tmp =pd .to_numeric (df_race [c ],errors ="coerce")
            if tmp .notna ().mean ()>0.95 :
                df_race [c ]=tmp 
            else :
                drop_obj .append (c )

    numeric_cols =[c for c in numeric_cols if c not in drop_obj ]

    df_race .to_parquet (outdir /"setup_race.parquet",index =False )

    features ={
    "id_cols":id_cols ,
    "categorical_cols":cat_cols ,
    "numeric_cols":numeric_cols ,
    "target_hints":["lap_time__mean","y_burn_race"],
    "notes":{
    "compound_source":f"{cfg.laps_table} -> compound_id_mode",
    "setup_source":f"{cfg.telemetry_table} -> first_valid per race",
    "dyn_source":f"{cfg.telemetry_table} -> mean/max per race",
    "dropped_object_cols":drop_obj ,
    }
    }
    _to_json (outdir /"features.json",features )

    print (f"[ok] feature_store gravado em: {outdir}")
    print (f"[ok] features.json ok | n_numeric={len(numeric_cols)} n_cat={len(cat_cols)}")

    return df_race 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (DEFAULT_SQLITE ))
    ap .add_argument ("--telemetry_table",default ="telemetry_flat")
    ap .add_argument ("--laps_table",default ="laps")
    ap .add_argument ("--out_table",default =DEFAULT_OUT_TABLE )
    ap .add_argument ("--feature_dir",default =str (DEFAULT_FEATURE_DIR ))
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    cfg =SetupRaceCfg (
    sqlite_path =args .sqlite_path ,
    telemetry_table =args .telemetry_table ,
    laps_table =args .laps_table ,
    out_table =args .out_table ,
    feature_dir =args .feature_dir ,
    drop_and_rebuild =args .drop_and_rebuild ,
    )
    build_setup_race_table (db_path =Path (cfg .sqlite_path ),cfg =cfg )


if __name__ =="__main__":
    main ()
