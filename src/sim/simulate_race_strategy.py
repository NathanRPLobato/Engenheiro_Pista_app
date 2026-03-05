# arquivo: src/sim/simulate_race_strategy.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple ,Optional 

import numpy as np 
import pandas as pd 

import tensorflow as tf 
from tensorflow import keras 


# =========================
# Config
# =========================
@dataclass 
class SimCfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"

    baselines_table :str ="baselines_track_compound"
    windows_table :str ="lap_degradation_windows"

    # modelos
    degr_model_dir :str ="models/champion/degradation_tcn"
    # risk_pace_dir: str = 'models/champion/risk_pace' # opcional, plugar depois

    # corrida
    trackId :str ="Montreal"
    n_laps :int =50 
    pit_time_s :float =22.0 

    # sequências
    max_windows_per_lap :int =60 
    min_windows_per_lap :int =8 

    # penalidades simples (calibráveis)
    fuel_penalty_s_per_lap :float =0.015 # ajuste fino depois
    tyre_age_penalty_s_per_lap :float =0.020 # ajuste fino depois

    # compostos a testar (IDs do teu dataset)
    compounds :List [int ]=None # se None, puxa do baseline do track


    # =========================
    # Utils DB
    # =========================
def _connect (db :str )->sqlite3 .Connection :
    return sqlite3 .connect (db )


def _read_baselines (con :sqlite3 .Connection ,table :str ,trackId :str )->pd .DataFrame :
    df =pd .read_sql (
    f"""
        SELECT trackId, compound_id, baseline_mean, baseline_median, baseline_std, n_laps, n_races
        FROM {table}
        WHERE trackId = ?
        """,
    con ,
    params =[str (trackId )],
    )
    df ["trackId"]=df ["trackId"].astype (str )
    df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce").astype ("Int64")
    df ["baseline_mean"]=pd .to_numeric (df ["baseline_mean"],errors ="coerce")
    df =df .dropna (subset =["compound_id","baseline_mean"]).copy ()
    df ["compound_id"]=df ["compound_id"].astype (int )
    return df 


def _load_features_spec (model_dir :Path )->Dict :
    meta_path =model_dir /"degradation_tcn.meta.json"
    if not meta_path .exists ():
        raise SystemExit (f"meta não existe: {meta_path}")
    return json .loads (meta_path .read_text (encoding ="utf-8"))


def _load_scaler (model_dir :Path )->Tuple [np .ndarray ,np .ndarray ,List [str ]]:
    sc_path =model_dir /"degradation_tcn.scalers.json"
    if not sc_path .exists ():
        raise SystemExit (f"scalers não existe: {sc_path}")
    j =json .loads (sc_path .read_text (encoding ="utf-8"))
    xs =j ["x_scaler"]
    mean =np .array (xs ["mean"],dtype =np .float32 )
    scale =np .array (xs ["scale"],dtype =np .float32 )
    x_cols =list (xs ["x_cols"])
    return mean ,scale ,x_cols 


def _scale_seq (X :np .ndarray ,mean :np .ndarray ,scale :np .ndarray )->np .ndarray :
# X: (T, F) ou (N, T, F)
    eps =1e-8 
    if X .ndim ==2 :
        return (X -mean )/(scale +eps )
    return (X -mean .reshape (1 ,1 ,-1 ))/(scale .reshape (1 ,1 ,-1 )+eps )


    # =========================
    # Prototype sequences
    # =========================
def _build_prototypes (
con :sqlite3 .Connection ,
windows_table :str ,
trackId :str ,
compounds :List [int ],
x_cols :List [str ],
max_windows_per_lap :int ,
min_windows_per_lap :int ,
)->Dict [Tuple [str ,int ],np .ndarray ]:
    """
    Cria um X_seq "típico" por (trackId, compound_id):
      - pega janelas reais
      - cria um índice de janela aproximado (rank por win_start_m dentro da volta)
      - agrega mediana por index
      - pad/trunca para T=max_windows_per_lap
    """
    cols_need =["race_id","trackId","lap_number","compound_id","win_start_m"]+x_cols 
    col_sql =", ".join ([f'"{c}"'for c in cols_need ])

    df =pd .read_sql (
    f"""
        SELECT {col_sql}
        FROM "{windows_table}"
        WHERE trackId = ?
          AND compound_id IN ({",".join(["?"] * len(compounds))})
        """,
    con ,
    params =[str (trackId )]+[int (c )for c in compounds ],
    )

    if df .empty :
        raise SystemExit ("Não achei janelas para esse track/compounds no lap_degradation_windows.")

    df ["trackId"]=df ["trackId"].astype (str )
    df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce")
    df ["lap_number"]=pd .to_numeric (df ["lap_number"],errors ="coerce")
    df ["win_start_m"]=pd .to_numeric (df ["win_start_m"],errors ="coerce")
    df =df .dropna (subset =["compound_id","lap_number","win_start_m"]).copy ()
    df ["compound_id"]=df ["compound_id"].astype (int )
    df ["lap_number"]=df ["lap_number"].astype (int )

    # numeric coerce
    for c in x_cols :
        df [c ]=pd .to_numeric (df [c ],errors ="coerce")

        # imputação global (mediana)
    med =df [x_cols ].median (numeric_only =True )
    df [x_cols ]=df [x_cols ].fillna (med ).fillna (0.0 )

    prototypes :Dict [Tuple [str ,int ],np .ndarray ]={}

    for comp in compounds :
        g =df [df ["compound_id"]==comp ].copy ()
        if g .empty :
            continue 

            # define window_index por volta: rank de win_start_m dentro (race_id,lap)
        g ["w_idx"]=(
        g .sort_values (["race_id","lap_number","win_start_m"])
        .groupby (["race_id","lap_number"])
        .cumcount ()
        )

        # filtra voltas com janelas suficientes
        counts =g .groupby (["race_id","lap_number"],as_index =False )["w_idx"].max ()
        counts ["n_win"]=counts ["w_idx"]+1 
        good =counts [counts ["n_win"]>=min_windows_per_lap ][["race_id","lap_number"]]
        g =g .merge (good ,on =["race_id","lap_number"],how ="inner")

        if g .empty :
            continue 

            # agrega por w_idx (mediana)
        agg =g .groupby ("w_idx",as_index =False )[x_cols ].median (numeric_only =True )

        # monta sequência
        T =max_windows_per_lap 
        F =len (x_cols )
        X =np .zeros ((T ,F ),dtype =np .float32 )

        for _ ,row in agg .iterrows ():
            idx =int (row ["w_idx"])
            if idx >=T :
                break 
            X [idx ]=row [x_cols ].to_numpy (dtype =np .float32 )

            # se primeiros índices vazios (caso raro), coloca mediana global
        if np .all (X .sum (axis =1 )==0 ):
            X [:]=med .to_numpy (dtype =np .float32 )

        prototypes [(str (trackId ),int (comp ))]=X 

    if not prototypes :
        raise SystemExit ("Não consegui criar prototypes. Cheque min_windows_per_lap / dados.")
    return prototypes 


    # =========================
    # Degradation predictor (TCN)
    # =========================
def _predict_lap_from_tcn (
model :keras .Model ,
X_seq :np .ndarray ,
mean :np .ndarray ,
scale :np .ndarray ,
)->Tuple [float ,float ]:
    """
    Retorna (pace_delta_lap_s, stress_lap)
    O modelo foi treinado em lap-level (média dos y por volta).
    Aqui: predição direta do modelo -> [y_pace_window, y_stress_window] agregados.
    """
    Xs =_scale_seq (X_seq ,mean ,scale ).astype (np .float32 )
    y =model .predict (Xs [np .newaxis ,...],verbose =0 ).ravel ()
    pace_delta =float (y [0 ])
    stress =float (y [1 ])
    return pace_delta ,stress 


    # =========================
    # Race Simulation
    # =========================
def _fuel_penalty (lap_idx :int ,n_laps :int ,cfg :SimCfg )->float :
# simples: penaliza mais no começo
    frac =1.0 -(lap_idx /max (1 ,n_laps -1 ))
    return cfg .fuel_penalty_s_per_lap *(0.5 +frac )


def _tyre_age_penalty (tyre_age :int ,cfg :SimCfg )->float :
# simples: cresce com idade
    return cfg .tyre_age_penalty_s_per_lap *float (tyre_age )


def simulate_strategy (
baseline_map :Dict [int ,float ],
prototypes :Dict [Tuple [str ,int ],np .ndarray ],
model :keras .Model ,
mean :np .ndarray ,
scale :np .ndarray ,
trackId :str ,
stint_compounds :List [int ],
stint_lengths :List [int ],
cfg :SimCfg ,
)->Dict :
    assert sum (stint_lengths )==cfg .n_laps 

    total =0.0 
    lap_times =[]
    pit_stops =0 
    lap_cursor =0 

    for si ,(comp ,L )in enumerate (zip (stint_compounds ,stint_lengths )):
        tyre_age =0 
        base =baseline_map [comp ]
        proto =prototypes [(trackId ,comp )]

        for k in range (L ):
            lap_idx =lap_cursor +k 

            # DL degradação (tcn)
            pace_delta ,stress =_predict_lap_from_tcn (model ,proto ,mean ,scale )

            # ajustes físicos simples
            fp =_fuel_penalty (lap_idx ,cfg .n_laps ,cfg )
            tp =_tyre_age_penalty (tyre_age ,cfg )

            # risk_pace (plugável) - por enquanto 0, você pluga o modelo depois
            risk_delta =0.0 

            lt =base +pace_delta +risk_delta +fp +tp 
            lap_times .append (lt )
            total +=lt 
            tyre_age +=1 

        lap_cursor +=L 

        # pit entre stints (exceto depois do último)
        if si <len (stint_compounds )-1 :
            total +=cfg .pit_time_s 
            pit_stops +=1 

    return {
    "trackId":trackId ,
    "n_laps":cfg .n_laps ,
    "compounds":stint_compounds ,
    "stints":stint_lengths ,
    "pit_stops":pit_stops ,
    "total_time_s":float (total ),
    "avg_lap_s":float (np .mean (lap_times )),
    "lap_times_s":lap_times ,
    }


def enumerate_strategies (compounds :List [int ],n_laps :int )->List [Tuple [List [int ],List [int ]]]:
    out =[]

    # 0 stop (1 stint)
    for c1 in compounds :
        out .append (([c1 ],[n_laps ]))

        # 1 stop (2 stints)
    for c1 in compounds :
        for c2 in compounds :
            for split in range (max (3 ,int (0.25 *n_laps )),min (n_laps -3 ,int (0.75 *n_laps ))+1 ):
                out .append (([c1 ,c2 ],[split ,n_laps -split ]))

                # 2 stops (3 stints)
    for c1 in compounds :
        for c2 in compounds :
            for c3 in compounds :
                for s1 in range (max (3 ,int (0.20 *n_laps )),min (n_laps -6 ,int (0.60 *n_laps ))+1 ):
                    for s2 in range (max (3 ,int (0.20 *n_laps )),min (n_laps -s1 -3 ,int (0.60 *n_laps ))+1 ):
                        s3 =n_laps -s1 -s2 
                        if s3 >=3 :
                            out .append (([c1 ,c2 ,c3 ],[s1 ,s2 ,s3 ]))

    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--trackId",required =True )
    ap .add_argument ("--n_laps",type =int ,required =True )
    ap .add_argument ("--pit_time_s",type =float ,default =22.0 )

    ap .add_argument ("--baselines_table",default ="baselines_track_compound")
    ap .add_argument ("--windows_table",default ="lap_degradation_windows")

    ap .add_argument ("--degr_model_dir",default ="models/champion/degradation_tcn")
    ap .add_argument ("--out_json",default ="data/sim_outputs/best_strategy.json")

    ap .add_argument ("--max_windows_per_lap",type =int ,default =60 )
    ap .add_argument ("--min_windows_per_lap",type =int ,default =8 )

    args =ap .parse_args ()

    cfg =SimCfg (
    sqlite_path =args .sqlite_path ,
    trackId =str (args .trackId ),
    n_laps =int (args .n_laps ),
    pit_time_s =float (args .pit_time_s ),
    baselines_table =str (args .baselines_table ),
    windows_table =str (args .windows_table ),
    degr_model_dir =str (args .degr_model_dir ),
    max_windows_per_lap =int (args .max_windows_per_lap ),
    min_windows_per_lap =int (args .min_windows_per_lap ),
    )

    out_path =Path (args .out_json )
    out_path .parent .mkdir (parents =True ,exist_ok =True )

    print ("\n=== SIMULATE RACE STRATEGY (baseline + TCN) ===")
    print (f"db={cfg.sqlite_path}")
    print (f"trackId={cfg.trackId} n_laps={cfg.n_laps} pit={cfg.pit_time_s}s")
    print (f"baselines={cfg.baselines_table} windows={cfg.windows_table}")
    print (f"model_dir={cfg.degr_model_dir}")

    con =_connect (cfg .sqlite_path )

    base_df =_read_baselines (con ,cfg .baselines_table ,cfg .trackId )
    if base_df .empty :
        con .close ()
        raise SystemExit ("Sem baseline pra essa pista. Rode build_baselines.")

    compounds =sorted (base_df ["compound_id"].unique ().tolist ())
    baseline_map ={int (r ["compound_id"]):float (r ["baseline_mean"])for _ ,r in base_df .iterrows ()}

    model_dir =Path (cfg .degr_model_dir )
    degr_model =keras .models .load_model (model_dir /"degradation_tcn.keras")
    mean ,scale ,x_cols =_load_scaler (model_dir )

    prototypes =_build_prototypes (
    con =con ,
    windows_table =cfg .windows_table ,
    trackId =cfg .trackId ,
    compounds =compounds ,
    x_cols =x_cols ,
    max_windows_per_lap =cfg .max_windows_per_lap ,
    min_windows_per_lap =cfg .min_windows_per_lap ,
    )

    con .close ()

    strategies =enumerate_strategies (compounds ,cfg .n_laps )

    best =None 
    for comps ,stints in strategies :
    # precisa de prototype pra todos
        ok =True 
        for c in comps :
            if (cfg .trackId ,c )not in prototypes :
                ok =False 
                break 
        if not ok :
            continue 

        res =simulate_strategy (
        baseline_map =baseline_map ,
        prototypes =prototypes ,
        model =degr_model ,
        mean =mean ,
        scale =scale ,
        trackId =cfg .trackId ,
        stint_compounds =comps ,
        stint_lengths =stints ,
        cfg =cfg ,
        )

        if best is None or res ["total_time_s"]<best ["total_time_s"]:
            best =res 

    if best is None :
        raise SystemExit ("Não achei nenhuma estratégia válida (prototypes faltando).")

    out_path .write_text (json .dumps (best ,indent =2 ),encoding ="utf-8")

    print ("\n=== BEST ===")
    print (f"compounds={best['compounds']} stints={best['stints']} stops={best['pit_stops']}")
    print (f"total_time_s={best['total_time_s']:.3f} avg_lap_s={best['avg_lap_s']:.3f}")
    print (f"saved: {out_path}")


if __name__ =="__main__":
    try :
        gpus =tf .config .list_physical_devices ("GPU")
        for g in gpus :
            tf .config .experimental .set_memory_growth (g ,True )
    except Exception :
        pass 

    main ()