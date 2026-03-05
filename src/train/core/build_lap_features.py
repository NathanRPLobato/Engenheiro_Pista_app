# arquivo: src/train/core/build_lap_features.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 

import numpy as np 
import pandas as pd 


SQLITE_PATH =Path ("data/refined/telemetry.sqlite")
FEATURE_STORE_DIR =Path ("data/feature_store/risk_pace")


@dataclass 
class OutlierCfg :
    mode :str # 'drop' ou 'clip'
    iqr_k :float 
    min_laps_group :int 


def _connect (db_path :Path )->sqlite3 .Connection :
    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    return con 


def _to_num (s :pd .Series )->pd .Series :
    if pd .api .types .is_numeric_dtype (s ):
        return s 
    return pd .to_numeric (s ,errors ="coerce")


def _safe_numeric (df :pd .DataFrame ,col :str ,min_ok_ratio :float =0.98 )->None :
    """
    Tenta converter colunas object para numérico.
    Se converter quase tudo, troca.
    Se virar muita coisa NaN, mantém como string.
    """
    if col not in df .columns :
        return 
    if pd .api .types .is_numeric_dtype (df [col ]):
        return 

        # Alguns pandas 3.x podem usar 'string[pyarrow]'
    col_dtype =str (df [col ].dtype ).lower ()
    if ("object"not in col_dtype )and ("string"not in col_dtype ):
        return 

    orig =df [col ]
    conv =pd .to_numeric (orig ,errors ="coerce")
    ok_ratio =np .isfinite (conv .to_numpy ()).mean ()if len (conv )else 0.0 

    if ok_ratio >=min_ok_ratio :
        df [col ]=conv .astype (np .float32 )
    else :
        df [col ]=orig .astype (str )


def _apply_outlier_filter (
df :pd .DataFrame ,
cfg :OutlierCfg ,
group_cols :list [str ],
target_col :str ,
)->tuple [pd .DataFrame ,pd .DataFrame ]:
    """
    Outlier por IQR dentro de cada grupo (trackId + compound_id).
    mode=drop: remove outliers
    mode=clip: corta para dentro de [q1-k*iqr, q3+k*iqr]
    """
    t0 =time .time ()

    df =df .reset_index (drop =True ).copy ()
    y =df [target_col ].to_numpy ()

    diag_rows =[]
    keep_mask =np .ones (len (df ),dtype =bool )

    for gkey ,g in df .groupby (group_cols ,dropna =False ):
        idx =g .index .to_numpy ()
        if len (idx )<cfg .min_laps_group :
            continue 

        vals =df .loc [idx ,target_col ].to_numpy ()
        vals =vals [np .isfinite (vals )]
        if len (vals )<cfg .min_laps_group :
            continue 

        q1 =np .quantile (vals ,0.25 )
        q3 =np .quantile (vals ,0.75 )
        iqr =q3 -q1 
        lo =q1 -cfg .iqr_k *iqr 
        hi =q3 +cfg .iqr_k *iqr 

        out =(y [idx ]<lo )|(y [idx ]>hi )

        row ={group_cols [0 ]:None ,group_cols [1 ]:None }
        if isinstance (gkey ,tuple ):
            for i ,c in enumerate (group_cols ):
                row [c ]=gkey [i ]
        else :
            row [group_cols [0 ]]=gkey 

        row .update (
        {
        "n":int (len (idx )),
        "q1":float (q1 ),
        "q3":float (q3 ),
        "iqr":float (iqr ),
        "lo":float (lo ),
        "hi":float (hi ),
        "n_out":int (out .sum ()),
        }
        )
        diag_rows .append (row )

        if cfg .mode =="drop":
            keep_mask [idx ]=keep_mask [idx ]&(~out )
        elif cfg .mode =="clip":
            df .loc [idx ,target_col ]=np .clip (df .loc [idx ,target_col ].to_numpy (),lo ,hi )
        else :
            raise ValueError (f"mode inválido: {cfg.mode}")

    df_out =df .loc [keep_mask ].reset_index (drop =True )
    diag =pd .DataFrame (diag_rows )

    t1 =time .time ()
    print (f"[ok] outlier_filter concluído em {t1 - t0:.2f}s  rows_before={len(df)} rows_after={len(df_out)}")
    return df_out ,diag 


def _agg_telemetry_per_lap (con :sqlite3 .Connection ,telemetry_table :str )->pd .DataFrame :
    """
    Agrega telemetry_flat por (race_id, lap_number) para gerar features por volta.
    """
    t0 =time .time ()

    cols =pd .read_sql (f"PRAGMA table_info('{telemetry_table}')",con )["name"].tolist ()
    cols_set =set (cols )

    def has (c :str )->bool :
        return c in cols_set 

    select_parts =["race_id","lap_number"]

    def add_avg (name :str ):
        select_parts .append (f"AVG({name}) AS {name}__mean")

    def add_max (name :str ):
        select_parts .append (f"MAX({name}) AS {name}__max")

    def add_min (name :str ):
        select_parts .append (f"MIN({name}) AS {name}__min")

    if has ("throttle"):
        add_avg ("throttle");add_max ("throttle")
    if has ("brake"):
        add_avg ("brake");add_max ("brake")
    if has ("steering"):
        select_parts .append ("AVG(ABS(steering)) AS steering_abs__mean")
        select_parts .append ("MAX(ABS(steering)) AS steering_abs__max")
    if has ("rpm"):
        add_avg ("rpm");add_max ("rpm")
    if has ("gear"):
        add_avg ("gear")

    if has ("speed"):
        add_avg ("speed");add_max ("speed")

    if has ("gforce_X"):
        add_avg ("gforce_X");add_max ("gforce_X")
    if has ("gforce_Y"):
        add_avg ("gforce_Y");add_max ("gforce_Y")

    if has ("pit_status"):
        add_avg ("pit_status");add_max ("pit_status")
    if has ("drs"):
        add_avg ("drs");add_max ("drs")

        # Fuel pode vir string, então CAST resolve
    if has ("fuel"):
        select_parts .append ("AVG(CAST(fuel AS REAL)) AS fuel__mean")
        select_parts .append ("MAX(CAST(fuel AS REAL)) AS fuel__max")
        select_parts .append ("MIN(CAST(fuel AS REAL)) AS fuel__min")

    sql =f"""
    SELECT
      {", ".join(select_parts)}
    FROM {telemetry_table}
    WHERE lap_number >= 0
    GROUP BY race_id, lap_number
    """

    df =pd .read_sql (sql ,con )

    t1 =time .time ()
    print (f"[ok] agregação {telemetry_table} por volta concluída em {t1 - t0:.2f}s  rows={len(df)}")
    return df 


def build_lap_features (
db_path :Path ,
laps_table :str ,
baselines_table :str ,
telemetry_table :str ,
out_table :str ,
outlier_cfg :OutlierCfg ,
write_parquet :bool ,
)->tuple [pd .DataFrame ,pd .DataFrame ]:
    con =_connect (db_path )

    laps =pd .read_sql (f"SELECT * FROM {laps_table};",con )
    base =pd .read_sql (f"SELECT * FROM {baselines_table};",con )

    # lap_time e fuel sempre numéricos
    laps ["lap_time"]=_to_num (laps ["lap_time"])
    if "fuel"in laps .columns :
        laps ["fuel"]=_to_num (laps ["fuel"])

        # agrega raw por volta
    agg =_agg_telemetry_per_lap (con ,telemetry_table )

    df =laps .merge (agg ,on =["race_id","lap_number"],how ="left")

    # junta baselines
    df =df .merge (
    base ,
    on =["trackId","compound_id"],
    how ="left",
    suffixes =("","__base"),
    )

    miss =df ["baseline_mean"].isna ().sum ()
    if miss >0 :
        print (f"[warn] {miss} voltas sem baseline_mean (track+compound sem baseline). Removendo.")
        df =df .loc [~df ["baseline_mean"].isna ()].reset_index (drop =True )

    df ["baseline_mean"]=_to_num (df ["baseline_mean"])
    df ["y_delta"]=df ["lap_time"]-df ["baseline_mean"]

    ok =np .isfinite (df ["y_delta"].to_numpy ())
    df =df .loc [ok ].reset_index (drop =True )

    df_f ,diag =_apply_outlier_filter (
    df =df ,
    cfg =outlier_cfg ,
    group_cols =["trackId","compound_id"],
    target_col ="y_delta",
    )

    # salva no sqlite
    df_f .to_sql (out_table ,con ,if_exists ="replace",index =False )
    diag .to_sql (f"{out_table}__outliers_diag",con ,if_exists ="replace",index =False )

    con .commit ()
    con .close ()

    if write_parquet :
        outdir =FEATURE_STORE_DIR 
        outdir .mkdir (parents =True ,exist_ok =True )

        # converte numericos safe (pandas 3.x não aceita errors='ignore')
        for c in df_f .columns :
            if c in ["race_id","trackId"]:
                continue 
            _safe_numeric (df_f ,c ,min_ok_ratio =0.98 )

            # garante que não tem object bizarro
        for c in df_f .columns :
            if df_f [c ].dtype =="object":
                df_f [c ]=df_f [c ].astype (str )

                # grava por corrida
        for rid ,g in df_f .groupby ("race_id"):
            outp =outdir /f"{rid}.parquet"
            g .to_parquet (outp ,index =False )

        feats ={
        "target":"y_delta",
        "id_cols":["race_id","lap_number"],
        "categorical":["trackId","compound_id"],
        "all_columns":df_f .columns .tolist (),
        }
        (outdir /"features.json").write_text (
        json .dumps (feats ,indent =2 ,ensure_ascii =False ),
        encoding ="utf-8",
        )
        print (f"[ok] parquets + features.json gravados em: {outdir}")

    return df_f ,diag 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--db",default =str (SQLITE_PATH ))
    ap .add_argument ("--laps_table",default ="laps")
    ap .add_argument ("--baselines_table",default ="baselines_track_compound")
    ap .add_argument ("--telemetry_table",default ="telemetry_flat")
    ap .add_argument ("--out_table",default ="lap_features_risk_pace")

    ap .add_argument ("--mode",choices =["drop","clip"],default ="drop")
    ap .add_argument ("--iqr_k",type =float ,default =1.5 )
    ap .add_argument ("--min_laps_group",type =int ,default =8 )

    ap .add_argument ("--write_parquet",action ="store_true")
    args =ap .parse_args ()

    print ("\n=== BUILD LAP FEATURES (RISK_PACE) ===")
    print (f"db={args.db}")
    print (f"laps_table={args.laps_table}")
    print (f"baselines_table={args.baselines_table}")
    print (f"telemetry_table={args.telemetry_table}")
    print (f"out_table={args.out_table}")
    print (f"outlier: mode={args.mode} iqr_k={args.iqr_k} min_group={args.min_laps_group}")

    cfg =OutlierCfg (mode =args .mode ,iqr_k =args .iqr_k ,min_laps_group =args .min_laps_group )

    df_f ,diag =build_lap_features (
    db_path =Path (args .db ),
    laps_table =args .laps_table ,
    baselines_table =args .baselines_table ,
    telemetry_table =args .telemetry_table ,
    out_table =args .out_table ,
    outlier_cfg =cfg ,
    write_parquet =args .write_parquet ,
    )

    print ("\n=== SUMMARY ===")
    print (pd .DataFrame ({"rows_out":[len (df_f )],"rows_diag":[len (diag )]}))


if __name__ =="__main__":
    main ()
