# arquivo: src/train/core/build_fingerprint_lap_table.py
from __future__ import annotations 

import argparse 
import sqlite3 
from pathlib import Path 

import numpy as np 
import pandas as pd 


DEFAULT_SQLITE =Path ("data/refined/telemetry.sqlite")
DEFAULT_EMB_WIN =Path ("models/champion/fingerprint_ae/fingerprint_embeddings_windows.parquet")
OUT_TABLE ="fingerprint_lap"


def _drop_view_or_table (con :sqlite3 .Connection ,name :str )->None :
    row =con .execute (
    "SELECT type FROM sqlite_master WHERE name=? COLLATE NOCASE",
    (name ,),
    ).fetchone ()
    if row is None :
        return 
    obj_type =(row [0 ]or "").lower ()
    if obj_type =="view":
        con .execute (f"DROP VIEW IF EXISTS {name};")
    else :
        con .execute (f"DROP TABLE IF EXISTS {name};")


def main ()->None :
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (DEFAULT_SQLITE ))
    ap .add_argument ("--embeddings_windows",default =str (DEFAULT_EMB_WIN ))
    ap .add_argument ("--out_table",default =OUT_TABLE )
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    ap .add_argument ("--min_windows_per_lap",type =int ,default =2 )
    args =ap .parse_args ()

    sqlite_path =Path (args .sqlite_path )
    emb_path =Path (args .embeddings_windows )

    if not sqlite_path .exists ():
        raise SystemExit (f"SQLite não encontrado: {sqlite_path}")
    if not emb_path .exists ():
        raise SystemExit (f"Embeddings windows não encontrado: {emb_path}")

    print ("\n=== BUILD FINGERPRINT PER LAP (from windows embeddings) ===")
    print (f"sqlite: {sqlite_path}")
    print (f"embeddings_windows: {emb_path}")
    print (f"out_table: {args.out_table}")
    print (f"min_windows_per_lap: {args.min_windows_per_lap}")

    dfw =pd .read_parquet (emb_path )

    required =["race_id","lap_number"]
    for c in required :
        if c not in dfw .columns :
            raise SystemExit (f"Coluna obrigatória não existe no parquet: {c}")

    fp_cols =[c for c in dfw .columns if c .startswith ("fp_")]
    if len (fp_cols )==0 :
        raise SystemExit ("Não achei colunas fp_XX no parquet. Era pra ter fp_00..fp_15.")

    dfw ["lap_number"]=pd .to_numeric (dfw ["lap_number"],errors ="coerce")
    dfw =dfw [np .isfinite (dfw ["lap_number"].values )]
    dfw ["lap_number"]=dfw ["lap_number"].astype (int )

    for c in fp_cols :
        dfw [c ]=pd .to_numeric (dfw [c ],errors ="coerce")

    g =dfw .groupby (["race_id","lap_number"],as_index =False )
    df_lap =g [fp_cols ].mean ()

    df_lap ["n_windows"]=g .size ()["size"].values 
    df_lap =df_lap [df_lap ["n_windows"]>=int (args .min_windows_per_lap )].reset_index (drop =True )

    print (f"[ok] laps com fingerprint: {len(df_lap)}")
    print (f"[ok] fp_dims: {len(fp_cols)}")

    con =sqlite3 .connect (sqlite_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")

    if args .drop_and_rebuild :
        _drop_view_or_table (con ,args .out_table )

    df_lap .to_sql (args .out_table ,con ,if_exists ="replace",index =False )

    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__race_lap ON {args.out_table}(race_id, lap_number);")
    con .commit ()

    n =con .execute (f"SELECT COUNT(*) FROM {args.out_table};").fetchone ()[0 ]
    print (f"[ok] gravado em SQLite: {args.out_table}  rows={n}")

    con .close ()


if __name__ =="__main__":
    main ()
