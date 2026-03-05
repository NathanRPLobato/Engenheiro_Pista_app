# arquivo: src/train/core/baselines.py
from __future__ import annotations 

import argparse 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Optional ,Tuple 

import numpy as np 
import pandas as pd 

DEFAULT_DB =Path ("data/refined/telemetry.sqlite")


@dataclass 
class BaselineConfig :
    last_n_races_per_track :int =3 
    min_laps_per_group :int =8 
    out_table :str ="baselines_track_compound"

    # opcional: filtrar só corridas secas se existir uma coluna booleana/numérica
    # ex: weather_is_wet, is_wet, rain, rain_percentage, track_wetness etc.
    dry_only :bool =False 
    wet_col :Optional [str ]=None # se None, tenta auto-detect


def _connect (db_path :Path ):
    import sqlite3 
    return sqlite3 .connect (str (db_path ))


def _extract_race_ts (race_id :str )->int :
    """
    race_id esperado: 'F12025-<epoch_ms>' ou algo com '-' antes do timestamp.
    Se não der pra parsear, cai num hash estável (pior caso).
    """
    try :
        s =str (race_id )
        if "-"in s :
            tail =s .split ("-")[-1 ]
            return int (tail )
            # fallback: tenta inteiro direto
        return int (s )
    except Exception :
    # fallback estável (não perfeito, mas evita crash)
        return abs (hash (str (race_id )))%(10 **12 )


def _detect_wet_col (con ,laps_table :str )->Optional [str ]:
    """
    Tenta achar uma coluna típica de chuva/molhado na tabela laps.
    Se não achar, retorna None.
    """
    cols =pd .read_sql (f"PRAGMA table_info({laps_table});",con )["name"].tolist ()
    candidates =[
    "is_wet","wet","weather_is_wet","track_wetness","rain","rain_percentage",
    "session_weather","weather","is_rain","rain_level"
    ]
    cols_lower ={c .lower ():c for c in cols }
    for cand in candidates :
        if cand .lower ()in cols_lower :
            return cols_lower [cand .lower ()]
    return None 


def _read_laps (con ,laps_table :str ,cfg :BaselineConfig )->pd .DataFrame :
    cols =pd .read_sql (f"PRAGMA table_info({laps_table});",con )["name"].tolist ()
    need ={"race_id","trackId","compound_id","lap_number","lap_time"}
    missing =[c for c in need if c not in cols ]
    if missing :
        raise SystemExit (f"Colunas obrigatórias ausentes em {laps_table}: {missing}")

    wet_col =cfg .wet_col 
    if cfg .dry_only and wet_col is None :
        wet_col =_detect_wet_col (con ,laps_table )

    select_cols =["race_id","trackId","compound_id","lap_number","lap_time"]
    if cfg .dry_only and wet_col is not None and wet_col in cols :
        select_cols .append (wet_col )

    df =pd .read_sql (
    f"SELECT {', '.join(select_cols)} FROM {laps_table};",
    con ,
    )

    # sane básico
    df ["race_id"]=df ["race_id"].astype (str )
    df ["trackId"]=df ["trackId"].astype (str )

    df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce")
    df ["lap_time"]=pd .to_numeric (df ["lap_time"],errors ="coerce")
    df ["lap_number"]=pd .to_numeric (df ["lap_number"],errors ="coerce")

    df =df .dropna (subset =["compound_id","lap_time","lap_number"])
    df ["compound_id"]=df ["compound_id"].astype (int )
    df ["lap_number"]=df ["lap_number"].astype (int )

    # remove lixo
    df =df [(df ["lap_number"]>=0 )&(df ["lap_time"]>0 )].copy ()

    # filtro seco (se possível)
    if cfg .dry_only and wet_col is not None and wet_col in df .columns :
        w =pd .to_numeric (df [wet_col ],errors ="coerce")
        # regra conservadora: seco = 0 / False / NaN tratado como seco
        dry_mask =(w .fillna (0 )<=0 )
        df =df .loc [dry_mask ].copy ()

        # timestamp numérico p/ ordenar corridas
    df ["race_ts"]=df ["race_id"].map (_extract_race_ts ).astype (np .int64 )

    return df 


def build_baselines (db_path :Path ,laps_table :str ,cfg :BaselineConfig )->pd .DataFrame :
    con =_connect (db_path )
    laps =_read_laps (con ,laps_table =laps_table ,cfg =cfg )

    if laps .empty :
        con .close ()
        return pd .DataFrame ()

        # 1) últimas N corridas por pista (por race_ts real)
    races_per_track =(
    laps [["trackId","race_id","race_ts"]]
    .drop_duplicates ()
    .sort_values (["trackId","race_ts"])
    )

    races_keep =(
    races_per_track .groupby ("trackId",as_index =False )
    .tail (cfg .last_n_races_per_track )
    .copy ()
    )

    # 2) filtra laps p/ só essas corridas
    laps_k =laps .merge (races_keep [["trackId","race_id"]],on =["trackId","race_id"],how ="inner")

    # 3) baseline por (trackId, compound_id)
    agg =(
    laps_k .groupby (["trackId","compound_id"],as_index =False )
    .agg (
    baseline_mean =("lap_time","mean"),
    baseline_median =("lap_time","median"),
    baseline_std =("lap_time","std"),
    n_laps =("lap_time","size"),
    n_races =("race_id","nunique"),
    last_race_ts =("race_ts","max"),
    )
    )

    # 4) filtra por mínimo de voltas por grupo
    agg =agg [agg ["n_laps"]>=cfg .min_laps_per_group ].copy ()

    # 5) audit: race_ids_used (TEXT) - NÃO é feature numérica
    used =(
    laps_k [["trackId","compound_id","race_id","race_ts"]]
    .drop_duplicates ()
    .sort_values (["trackId","compound_id","race_ts"])
    .groupby (["trackId","compound_id"],as_index =False )
    .agg (race_ids_used =("race_id",lambda s :",".join (s .astype (str ).tolist ())))
    )

    out =agg .merge (used ,on =["trackId","compound_id"],how ="left")

    # 6) salva no SQLite
    out .to_sql (cfg .out_table ,con ,if_exists ="replace",index =False )
    con .close ()
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--db",default =str (DEFAULT_DB ))
    ap .add_argument ("--laps_table",default ="laps")
    ap .add_argument ("--last_n_races",type =int ,default =3 )
    ap .add_argument ("--min_laps",type =int ,default =8 )
    ap .add_argument ("--out_table",default ="baselines_track_compound")

    ap .add_argument ("--dry_only",action ="store_true")
    ap .add_argument ("--wet_col",default =None )

    args =ap .parse_args ()

    cfg =BaselineConfig (
    last_n_races_per_track =int (args .last_n_races ),
    min_laps_per_group =int (args .min_laps ),
    out_table =str (args .out_table ),
    dry_only =bool (args .dry_only ),
    wet_col =str (args .wet_col )if args .wet_col else None ,
    )

    print ("\n=== BUILD BASELINES (track + compound) ===")
    print (f"db={args.db}")
    print (f"laps_table={args.laps_table}")
    print (f"last_n_races_per_track={cfg.last_n_races_per_track}")
    print (f"min_laps_per_group={cfg.min_laps_per_group}")
    print (f"dry_only={cfg.dry_only} wet_col={cfg.wet_col}")
    print (f"out_table={cfg.out_table}")

    df =build_baselines (db_path =Path (args .db ),laps_table =args .laps_table ,cfg =cfg )

    print ("\n=== BASELINES SUMMARY ===")
    if df .empty :
        print ("Baselines ficou vazio. (min_laps alto demais, ou filtro seco removeu tudo, ou faltam dados.)")
    else :
        print (df .head (20 ).to_string (index =False ))
        print ("\nTop grupos por n_laps:")
        print (df .sort_values ("n_laps",ascending =False ).head (10 ).to_string (index =False ))

    print ("\nOK. Baselines gravado na tabela:",cfg .out_table )


if __name__ =="__main__":
    main ()