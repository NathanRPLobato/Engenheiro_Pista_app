# arquivo: src/train/core/merge_fingerprint_into_risk_pace.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
from pathlib import Path 

import numpy as np 
import pandas as pd 


DEFAULT_SQLITE =Path ("data/refined/telemetry.sqlite")


def _ensure_dir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _safe_numeric (df :pd .DataFrame ,cols :list [str ])->None :
    for c in cols :
        if c in df .columns :
            df [c ]=pd .to_numeric (df [c ],errors ="coerce")


def main ()->None :
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (DEFAULT_SQLITE ))
    ap .add_argument ("--lap_features_table",default ="lap_features_risk_pace")
    ap .add_argument ("--fingerprint_table",default ="fingerprint_lap")
    ap .add_argument ("--out_table",default ="lap_features_risk_pace_fp")
    ap .add_argument ("--out_dir",default ="data/feature_store/risk_pace_fp")
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    sqlite_path =Path (args .sqlite_path )
    out_dir =Path (args .out_dir )
    _ensure_dir (out_dir )

    if not sqlite_path .exists ():
        raise SystemExit (f"SQLite não encontrado: {sqlite_path}")

    print ("\n=== MERGE FINGERPRINT -> RISK_PACE FEATURES ===")
    print (f"sqlite: {sqlite_path}")
    print (f"lap_features_table: {args.lap_features_table}")
    print (f"fingerprint_table: {args.fingerprint_table}")
    print (f"out_table: {args.out_table}")
    print (f"out_dir: {out_dir}")

    con =sqlite3 .connect (sqlite_path )

    lf =pd .read_sql (f"SELECT * FROM {args.lap_features_table};",con )
    fp =pd .read_sql (f"SELECT * FROM {args.fingerprint_table};",con )

    if len (lf )==0 :
        raise SystemExit ("lap_features está vazia. Primeiro rode build_lap_features.")
    if len (fp )==0 :
        raise SystemExit ("fingerprint_lap está vazia. Primeiro gere fingerprint por volta.")

    if "race_id"not in lf .columns or "lap_number"not in lf .columns :
        raise SystemExit ("lap_features precisa ter race_id e lap_number.")
    if "race_id"not in fp .columns or "lap_number"not in fp .columns :
        raise SystemExit ("fingerprint_lap precisa ter race_id e lap_number.")

    fp_cols =[c for c in fp .columns if c .startswith ("fp_")]
    if len (fp_cols )==0 :
        raise SystemExit ("fingerprint_lap não tem fp_*. Algo deu errado no build_fingerprint_lap_table.")

    lf ["lap_number"]=pd .to_numeric (lf ["lap_number"],errors ="coerce")
    fp ["lap_number"]=pd .to_numeric (fp ["lap_number"],errors ="coerce")
    lf =lf [np .isfinite (lf ["lap_number"].values )].copy ()
    fp =fp [np .isfinite (fp ["lap_number"].values )].copy ()
    lf ["lap_number"]=lf ["lap_number"].astype (int )
    fp ["lap_number"]=fp ["lap_number"].astype (int )

    for c in fp_cols :
        fp [c ]=pd .to_numeric (fp [c ],errors ="coerce")

    df =lf .merge (fp [["race_id","lap_number"]+fp_cols ],on =["race_id","lap_number"],how ="left")

    miss =int (df [fp_cols [0 ]].isna ().sum ())
    print (f"[diag] laps sem fingerprint após merge: {miss}/{len(df)}")

    df [fp_cols ]=df [fp_cols ].fillna (0.0 )

    _safe_numeric (
    df ,
    [
    "lap_time",
    "fuel",
    "baseline_mean",
    "baseline_median",
    "baseline_std",
    "baseline_n_laps",
    "baseline_n_races",
    "y_delta",
    "compound_id",
    ],
    )

    if args .drop_and_rebuild :
        con .execute (f"DROP TABLE IF EXISTS {args.out_table};")
        con .commit ()

    df .to_sql (args .out_table ,con ,if_exists ="replace",index =False )
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__race_lap ON {args.out_table}(race_id, lap_number);")
    con .commit ()

    print (f"[ok] SQLite table criada: {args.out_table}  rows={len(df)}")

    # escreve feature store por corrida
    race_groups =df .groupby ("race_id",as_index =False )
    for rid ,g in race_groups :
        outp =out_dir /f"{rid}.parquet"
        g .to_parquet (outp ,index =False )

        # features.json
        # regra: trackId e compound_id são categóricas
    non_features =set (["y_delta","race_id"])
    categorical =[c for c in ["trackId","compound_id"]if c in df .columns ]

    feature_candidates =[c for c in df .columns if c not in non_features ]
    feature_candidates =[c for c in feature_candidates if c not in ["lap_end_row_id","race_ids_used"]]

    numeric_features =[]
    for c in feature_candidates :
        if c in categorical :
            continue 
        if c =="lap_number":
            numeric_features .append (c )
            continue 
        if pd .api .types .is_numeric_dtype (df [c ]):
            numeric_features .append (c )

    features_payload ={
    "target":"y_delta",
    "categorical":categorical ,
    "numeric":sorted (numeric_features ),
    "all_features":sorted (categorical +numeric_features ),
    "fingerprint_cols":sorted (fp_cols ),
    }

    (out_dir /"features.json").write_text (json .dumps (features_payload ,indent =2 ),encoding ="utf-8")

    print (f"[ok] feature_store gravado em: {out_dir}")
    print (f"[ok] features.json ok | n_numeric={len(features_payload['numeric'])} n_cat={len(categorical)} n_fp={len(fp_cols)}")

    con .close ()


if __name__ =="__main__":
    main ()
