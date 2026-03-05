# arquivo: src/train/core/build_risk_pace_priors.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple 

import numpy as np 
import pandas as pd 


DEFAULT_DB =Path ("data/refined/telemetry.sqlite")


@dataclass 
class Cfg :
    sqlite_path :str =str (DEFAULT_DB )
    risk_table :str ="lap_features_risk_pace"
    out_table :str ="risk_pace_priors_track_compound"
    last_n_races_per_track :int =3 
    drop_and_rebuild :bool =False 
    risk_model_dir :str ="models/champion/risk_pace"


def _connect (db :str )->sqlite3 .Connection :
    con =sqlite3 .connect (db )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    con .execute ("PRAGMA temp_store=MEMORY;")
    return con 


def _extract_race_ts (race_id :str )->int :
    try :
        s =str (race_id )
        if "-"in s :
            return int (s .split ("-")[-1 ])
        return int (s )
    except Exception :
        return abs (hash (str (race_id )))%(10 **12 )


def _load_model_features (risk_model_dir :str )->Tuple [List [str ],List [str ]]:
    d =Path (risk_model_dir )
    feat_path =d /"lgbm__risk_pace.features.json"
    if not feat_path .exists ():
        raise SystemExit (f"features.json do risk_pace não existe em: {feat_path}")

    j =json .loads (feat_path .read_text (encoding ="utf-8"))
    features =list (j .get ("features",[]))
    categorical =list (j .get ("categorical",[]))
    if not features :
        raise SystemExit ("features.json do risk_pace não tem chave 'features' ou está vazio.")
    return features ,categorical 


def _get_table_cols (con :sqlite3 .Connection ,table :str )->List [str ]:
    return pd .read_sql (f"PRAGMA table_info({table});",con )["name"].tolist ()


def _pick_stable_features (model_features :List [str ])->List [str ]:
    """
    Priors = coisas estáveis por pista+composto.
    Remove lap-level e IDs.
    """
    drop ={
    "race_id",
    "lap_number",
    "lap_time",
    "fuel",# fuel é lap-level, a gente passa na simulação
    "trackId",# chave, não prior numérico
    "compound_id",# chave
    }
    # mantém tudo que não é lap-level
    stable =[f for f in model_features if f not in drop ]
    return stable 


def build_priors (cfg :Cfg )->pd .DataFrame :
    con =_connect (cfg .sqlite_path )

    model_features ,_ =_load_model_features (cfg .risk_model_dir )

    cols =_get_table_cols (con ,cfg .risk_table )
    need_base =["race_id","trackId","compound_id"]
    missing_base =[c for c in need_base if c not in cols ]
    if missing_base :
        con .close ()
        raise SystemExit (f"Faltando colunas base em {cfg.risk_table}: {missing_base}")

    stable_feats =_pick_stable_features (model_features )

    # Vamos pegar só as features do modelo que EXISTEM na tabela
    stable_in_table =[f for f in stable_feats if f in cols ]

    # Também precisamos de lap_time/fuel/lap_number Não p/ priors.
    select_cols =["race_id","trackId","compound_id"]+stable_in_table 

    df =pd .read_sql (
    f"SELECT {', '.join(select_cols)} FROM {cfg.risk_table};",
    con ,
    )

    if df .empty :
        con .close ()
        raise SystemExit (f"Tabela {cfg.risk_table} vazia.")

        # sane
    df ["race_id"]=df ["race_id"].astype (str )
    df ["trackId"]=df ["trackId"].astype (str )
    df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce").astype ("Int64")
    df =df .dropna (subset =["compound_id"]).copy ()
    df ["compound_id"]=df ["compound_id"].astype (int )

    df ["race_ts"]=df ["race_id"].map (_extract_race_ts ).astype (np .int64 )

    # últimas N corridas por pista
    races =(
    df [["trackId","race_id","race_ts"]]
    .drop_duplicates ()
    .sort_values (["trackId","race_ts"])
    )
    keep =races .groupby ("trackId",as_index =False ).tail (cfg .last_n_races_per_track )
    df =df .merge (keep [["trackId","race_id"]],on =["trackId","race_id"],how ="inner")

    # converte numérico nas features disponíveis
    for c in stable_in_table :
        df [c ]=pd .to_numeric (df [c ],errors ="coerce")

        # agrega
    agg_dict ={}
    for c in stable_in_table :
        agg_dict [f"{c}__median"]=(c ,"median")

    out =(
    df .groupby (["trackId","compound_id"],as_index =False )
    .agg (**agg_dict ,rows =("race_id","size"),n_races_used =("race_id","nunique"))
    )

    if cfg .drop_and_rebuild :
        con .execute (f"DROP TABLE IF EXISTS {cfg.out_table};")

    out .to_sql (cfg .out_table ,con ,if_exists ="replace",index =False )
    con .close ()

    # log útil
    missing_from_table =[f for f in stable_feats if f not in cols ]
    print (f"[info] model_features_total={len(model_features)}")
    print (f"[info] stable_features={len(stable_feats)}")
    print (f"[info] stable_in_table={len(stable_in_table)}")
    if missing_from_table :
        print (f"[warn] stable features que o modelo pede mas NÃO estão na tabela ({len(missing_from_table)}):")
        print ("       "+", ".join (missing_from_table [:50 ])+(" ..."if len (missing_from_table )>50 else ""))

    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (DEFAULT_DB ))
    ap .add_argument ("--risk_table",default ="lap_features_risk_pace")
    ap .add_argument ("--out_table",default ="risk_pace_priors_track_compound")
    ap .add_argument ("--last_n_races_per_track",type =int ,default =3 )
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    ap .add_argument ("--risk_model_dir",default ="models/champion/risk_pace")
    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    risk_table =args .risk_table ,
    out_table =args .out_table ,
    last_n_races_per_track =int (args .last_n_races_per_track ),
    drop_and_rebuild =bool (args .drop_and_rebuild ),
    risk_model_dir =str (args .risk_model_dir ),
    )

    print ("\n=== BUILD RISK PACE PRIORS (AUTO-DETECT) ===")
    print (f"sqlite={cfg.sqlite_path}")
    print (f"risk_table={cfg.risk_table}")
    print (f"out_table={cfg.out_table}")
    print (f"last_n_races_per_track={cfg.last_n_races_per_track}")
    print (f"risk_model_dir={cfg.risk_model_dir}")
    print (f"drop_and_rebuild={cfg.drop_and_rebuild}")

    df =build_priors (cfg )
    print (f"[ok] priors rows={len(df)}")
    print (df .head (10 ).to_string (index =False ))


if __name__ =="__main__":
    main ()