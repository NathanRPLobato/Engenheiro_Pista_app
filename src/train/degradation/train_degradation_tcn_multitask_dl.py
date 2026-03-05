# arquivo: src/train/degradation/train_degradation_tcn_multitask_dl.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple ,Optional 

import numpy as np 
import pandas as pd 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow .keras import layers 

from sklearn .model_selection import GroupShuffleSplit 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import mean_absolute_error ,mean_squared_error ,r2_score 


# =========================
# Spec (contrato do dataset)
# =========================
DEFAULT_SPEC ={
"table":"lap_degradation_windows",
"id_cols":[
"race_id","trackId","lap_number","compound_id",
"win_start_m","win_end_m","window_m","stride_m","n_samples"
],
"x_cols":[
"throttle__mean","throttle__var","throttle__max","throttle__min",
"brake__mean","brake__var","brake__max","brake__min",
"steering__mean","steering__var","steering__max","steering__min",
"fuel__mean",
"gforce_X__mean","gforce_Y__mean","angular_vel_Z__mean",
"slip_abs__mean","slip_abs__max",
"tyre_temp_mean","tyre_press_mean",
"speed2__mean",
"dt_window","lap_dist_max",
"baseline_mean_tc","baseline_mean_track","baseline_mean",
"baseline_window_time"
],
"y_cols":["y_pace_window","y_stress_window"],
}


# =========================
# Config
# =========================
@dataclass 
class Cfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"
    table :str ="lap_degradation_windows"
    spec_json :Optional [str ]=None # caminho pro JSON acima

    outdir :str ="models/champion/degradation_tcn"
    feature_dir :str ="data/feature_store/degradation_windows"

    seed :int =42 
    test_size :float =0.20 

    # sequência
    max_windows_per_lap :int =60 
    min_windows_per_lap :int =8 

    # treino
    epochs :int =300 
    batch_size :int =256 
    lr :float =1e-3 
    patience :int =60 

    # TCN
    n_blocks :int =6 
    n_filters :int =96 
    kernel_size :int =3 
    dropout :float =0.15 
    l2 :float =1e-6 


    # =========================
    # Utils
    # =========================
def _ensure_dir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _connect (db_path :str )->sqlite3 .Connection :
    return sqlite3 .connect (db_path )


def _table_cols (con :sqlite3 .Connection ,table :str )->List [str ]:
    return pd .read_sql (f"PRAGMA table_info({table});",con )["name"].tolist ()


def _read_table (con :sqlite3 .Connection ,table :str ,cols :List [str ])->pd .DataFrame :
    col_sql =", ".join ([f'"{c}"'for c in cols ])
    return pd .read_sql (f'SELECT {col_sql} FROM "{table}";',con )


def _load_spec (path :Optional [str ],table_fallback :str )->Dict :
    if path is None :
        spec =dict (DEFAULT_SPEC )
        spec ["table"]=table_fallback 
        return spec 

    p =Path (path )
    if not p .exists ():
        raise SystemExit (f"spec_json não existe: {path}")

    j =json .loads (p .read_text (encoding ="utf-8"))
    if "x_cols"not in j or "y_cols"not in j or "id_cols"not in j :
        raise SystemExit ("spec_json inválido: precisa ter id_cols, x_cols, y_cols.")
    if "table"not in j :
        j ["table"]=table_fallback 
    return j 


def _coerce_numeric (df :pd .DataFrame ,cols :List [str ])->pd .DataFrame :
    out =df .copy ()
    for c in cols :
        out [c ]=pd .to_numeric (out [c ],errors ="coerce")
    return out 


def _metrics (y_true :np .ndarray ,y_pred :np .ndarray ,y_cols :List [str ])->Dict :
    out ={}
    for j ,name in enumerate (y_cols ):
        yt =y_true [:,j ]
        yp =y_pred [:,j ]
        out [name ]={
        "mae":float (mean_absolute_error (yt ,yp )),
        "rmse":float (np .sqrt (mean_squared_error (yt ,yp ))),
        "r2":float (r2_score (yt ,yp )),
        }
    out ["macro"]={
    "mae":float (np .mean ([out [k ]["mae"]for k in y_cols ])),
    "rmse":float (np .mean ([out [k ]["rmse"]for k in y_cols ])),
    "r2":float (np .mean ([out [k ]["r2"]for k in y_cols ])),
    }
    return out 


    # =========================
    # Model (TCN)
    # =========================
def _tcn_block (x ,n_filters :int ,kernel_size :int ,dilation :int ,dropout :float ,l2 :float ):
    reg =keras .regularizers .l2 (l2 )if l2 and l2 >0 else None 

    h =layers .Conv1D (
    filters =n_filters ,
    kernel_size =kernel_size ,
    padding ="causal",
    dilation_rate =dilation ,
    activation ="relu",
    kernel_regularizer =reg ,
    )(x )
    h =layers .Dropout (dropout )(h )

    h =layers .Conv1D (
    filters =n_filters ,
    kernel_size =kernel_size ,
    padding ="causal",
    dilation_rate =dilation ,
    activation ="relu",
    kernel_regularizer =reg ,
    )(h )
    h =layers .Dropout (dropout )(h )

    if x .shape [-1 ]!=n_filters :
        x =layers .Conv1D (filters =n_filters ,kernel_size =1 ,padding ="same")(x )

    out =layers .Add ()([x ,h ])
    out =layers .LayerNormalization ()(out )
    return out 


def _build_model (T :int ,F :int ,n_out :int ,cfg :Cfg )->keras .Model :
    inp =keras .Input (shape =(T ,F ),name ="x_seq")

    x =inp 
    for b in range (cfg .n_blocks ):
        dilation =2 **b 
        x =_tcn_block (
        x ,
        n_filters =cfg .n_filters ,
        kernel_size =cfg .kernel_size ,
        dilation =dilation ,
        dropout =cfg .dropout ,
        l2 =cfg .l2 ,
        )

    x =layers .GlobalAveragePooling1D ()(x )
    x =layers .Dense (128 ,activation ="relu")(x )
    x =layers .Dropout (cfg .dropout )(x )
    out =layers .Dense (n_out ,activation ="linear",name ="y")(x )

    model =keras .Model (inp ,out )
    model .compile (
    optimizer =keras .optimizers .Adam (learning_rate =cfg .lr ),
    loss ="huber",
    metrics =[keras .metrics .MeanAbsoluteError (name ="mae")],
    )
    return model 


    # =========================
    # Sequence builder
    # =========================
def _build_sequences_from_spec (
df :pd .DataFrame ,
id_cols :List [str ],
x_cols :List [str ],
y_cols :List [str ],
cfg :Cfg 
)->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """
    Retorna:
      X_seq: (N_laps, T, F) com padding/trunc
      Y:     (N_laps, n_targets) -> média do target na volta
      groups:(N_laps,) race_id (group split)
    """
    need =set (["race_id","lap_number","win_start_m"])
    missing =[c for c in need if c not in df .columns ]
    if missing :
        raise SystemExit (f"Faltando colunas obrigatórias pra sequência: {missing}")

    df =df .copy ()
    df ["race_id"]=df ["race_id"].astype (str )
    df ["lap_number"]=pd .to_numeric (df ["lap_number"],errors ="coerce")
    df ["win_start_m"]=pd .to_numeric (df ["win_start_m"],errors ="coerce")
    df =df .dropna (subset =["lap_number","win_start_m"]).copy ()
    df ["lap_number"]=df ["lap_number"].astype (int )

    # numéricos
    df =_coerce_numeric (df ,x_cols +y_cols )

    # imputação: mediana global por feature
    med =df [x_cols ].median (numeric_only =True )
    df [x_cols ]=df [x_cols ].fillna (med ).fillna (0.0 )

    # targets: drop NaN (não dá pra treinar com y faltando)
    df =df .dropna (subset =y_cols ).copy ()

    df =df .sort_values (["race_id","lap_number","win_start_m"]).reset_index (drop =True )

    X_list =[]
    Y_list =[]
    groups_list =[]

    T =cfg .max_windows_per_lap 
    F =len (x_cols )

    for (race_id ,lapn ),g in df .groupby (["race_id","lap_number"],sort =False ):
        n =len (g )
        if n <cfg .min_windows_per_lap :
            continue 

        Xi =g [x_cols ].to_numpy (dtype =np .float32 )

        if n >=T :
            Xi2 =Xi [:T ]
        else :
            pad =np .zeros ((T -n ,F ),dtype =np .float32 )
            Xi2 =np .vstack ([Xi ,pad ])

        Yi =g [y_cols ].to_numpy (dtype =np .float32 )
        y_lap =Yi .mean (axis =0 )

        X_list .append (Xi2 )
        Y_list .append (y_lap )
        groups_list .append (race_id )

    if not X_list :
        raise SystemExit ("Não consegui montar sequências. Baixe min_windows_per_lap ou verifique o dataset.")

    X_seq =np .stack (X_list ,axis =0 )
    Y =np .stack (Y_list ,axis =0 )
    groups =np .array (groups_list ,dtype =object )
    return X_seq ,Y ,groups 


def main ():
    ap =argparse .ArgumentParser ()

    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--table",default ="lap_degradation_windows")
    ap .add_argument ("--spec_json",default =None )

    ap .add_argument ("--feature_dir",default ="data/feature_store/degradation_windows")
    ap .add_argument ("--outdir",default ="models/champion/degradation_tcn")

    ap .add_argument ("--max_windows_per_lap",type =int ,default =60 )
    ap .add_argument ("--min_windows_per_lap",type =int ,default =8 )

    ap .add_argument ("--epochs",type =int ,default =300 )
    ap .add_argument ("--batch_size",type =int ,default =256 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--seed",type =int ,default =42 )
    ap .add_argument ("--test_size",type =float ,default =0.20 )

    ap .add_argument ("--n_blocks",type =int ,default =6 )
    ap .add_argument ("--n_filters",type =int ,default =96 )
    ap .add_argument ("--kernel_size",type =int ,default =3 )
    ap .add_argument ("--dropout",type =float ,default =0.15 )
    ap .add_argument ("--l2",type =float ,default =1e-6 )
    ap .add_argument ("--patience",type =int ,default =60 )

    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    table =args .table ,
    spec_json =args .spec_json ,
    feature_dir =args .feature_dir ,
    outdir =args .outdir ,
    max_windows_per_lap =args .max_windows_per_lap ,
    min_windows_per_lap =args .min_windows_per_lap ,
    epochs =args .epochs ,
    batch_size =args .batch_size ,
    lr =args .lr ,
    seed =args .seed ,
    test_size =args .test_size ,
    n_blocks =args .n_blocks ,
    n_filters =args .n_filters ,
    kernel_size =args .kernel_size ,
    dropout =args .dropout ,
    l2 =args .l2 ,
    patience =args .patience ,
    )

    outdir =Path (cfg .outdir )
    feature_dir =Path (cfg .feature_dir )
    _ensure_dir (outdir )
    _ensure_dir (feature_dir )

    np .random .seed (cfg .seed )
    tf .random .set_seed (cfg .seed )

    spec =_load_spec (cfg .spec_json ,table_fallback =cfg .table )
    table =spec .get ("table",cfg .table )
    id_cols =list (spec ["id_cols"])
    x_cols =list (spec ["x_cols"])
    y_cols =list (spec ["y_cols"])

    print ("\n=== TRAIN DEGRADATION TCN (SPEC-DRIVEN, NO GUESSING) ===")
    print (f"sqlite={cfg.sqlite_path}")
    print (f"table={table}")
    print (f"outdir={cfg.outdir}")
    print (f"feature_dir={cfg.feature_dir}")
    print (f"seq: max_windows_per_lap={cfg.max_windows_per_lap} min_windows_per_lap={cfg.min_windows_per_lap}")
    print (f"X={len(x_cols)} cols  Y={len(y_cols)} cols")

    t0 =time .time ()

    con =_connect (cfg .sqlite_path )
    existing =set (_table_cols (con ,table ))
    if not existing :
        con .close ()
        raise SystemExit (f"Tabela não existe: {table}")

        # valida colunas
    need =set (id_cols +x_cols +y_cols )
    miss =sorted ([c for c in need if c not in existing ])
    if miss :
        con .close ()
        raise SystemExit (f"Dataset não bate com SPEC. Colunas faltando: {miss}")

        # lê só o necessário (bem mais rápido e sem lixo)
    df =_read_table (con ,table ,cols =sorted (list (need )))
    con .close ()

    if df .empty :
        raise SystemExit ("Tabela vazia. Rode o build_lap_degradation_windows primeiro.")

        # constrói sequências por volta
    X_seq ,Y ,groups =_build_sequences_from_spec (df ,id_cols =id_cols ,x_cols =x_cols ,y_cols =y_cols ,cfg =cfg )
    N ,T ,F =X_seq .shape 
    print (f"[ok] sequences: N={N} T={T} F={F} targets={Y.shape[1]}")

    # normalize X (fit no train depois do split)
    gss =GroupShuffleSplit (n_splits =1 ,test_size =cfg .test_size ,random_state =cfg .seed )
    tr_idx ,te_idx =next (gss .split (X_seq ,Y ,groups =groups ))

    Xtr ,Xte =X_seq [tr_idx ],X_seq [te_idx ]
    ytr ,yte =Y [tr_idx ],Y [te_idx ]
    gtr ,gte =groups [tr_idx ],groups [te_idx ]

    # fit scaler no train (flatten)
    scaler =StandardScaler ()
    Xtr_flat =Xtr .reshape (len (Xtr )*T ,F ).astype (np .float32 )
    Xte_flat =Xte .reshape (len (Xte )*T ,F ).astype (np .float32 )

    Xtr_s =scaler .fit_transform (Xtr_flat ).reshape (len (Xtr ),T ,F ).astype (np .float32 )
    Xte_s =scaler .transform (Xte_flat ).reshape (len (Xte ),T ,F ).astype (np .float32 )

    print (f"[ok] split: train={len(tr_idx)} test={len(te_idx)} races_train={len(set(gtr))} races_test={len(set(gte))}")

    # model
    model =_build_model (T =T ,F =F ,n_out =len (y_cols ),cfg =cfg )

    cb =[
    keras .callbacks .EarlyStopping (monitor ="val_loss",patience =cfg .patience ,restore_best_weights =True ),
    keras .callbacks .ReduceLROnPlateau (monitor ="val_loss",factor =0.5 ,patience =15 ,min_lr =1e-5 ),
    ]

    model .fit (
    Xtr_s ,ytr ,
    validation_data =(Xte_s ,yte ),
    epochs =cfg .epochs ,
    batch_size =cfg .batch_size ,
    verbose =1 ,
    callbacks =cb ,
    )

    ypred =model .predict (Xte_s ,batch_size =1024 ,verbose =0 )
    metrics =_metrics (yte ,ypred ,y_cols )

    print ("\n=== METRICS PER TARGET (lap-level) ===")
    for k in y_cols :
        print (f"{k:16s} MAE={metrics[k]['mae']:.4f} RMSE={metrics[k]['rmse']:.4f} R2={metrics[k]['r2']:.4f}")
    print ("\nMetrics (macro):",metrics ["macro"])

    # salva modelo
    model .save (outdir /"degradation_tcn.keras")

    # save scaler + spec/meta (pra API e pra inferência sem erro)
    scalers ={
    "x_scaler":{
    "mean":scaler .mean_ .tolist (),
    "scale":scaler .scale_ .tolist (),
    "var":scaler .var_ .tolist (),
    "n_features":int (len (scaler .mean_ )),
    "x_cols":x_cols ,
    }
    }
    (outdir /"degradation_tcn.scalers.json").write_text (json .dumps (scalers ,indent =2 ),encoding ="utf-8")

    meta ={
    "sqlite_path":cfg .sqlite_path ,
    "table":table ,
    "id_cols":id_cols ,
    "x_cols":x_cols ,
    "y_cols":y_cols ,
    "max_windows_per_lap":cfg .max_windows_per_lap ,
    "min_windows_per_lap":cfg .min_windows_per_lap ,
    "seed":cfg .seed ,
    "test_size":cfg .test_size ,
    "n_sequences":int (N ),
    "notes":spec .get ("notes",{}),
    }
    (outdir /"degradation_tcn.meta.json").write_text (json .dumps (meta ,indent =2 ),encoding ="utf-8")
    (outdir /"degradation_tcn.metrics.json").write_text (json .dumps (metrics ,indent =2 ),encoding ="utf-8")

    # salva features.json no feature_store também
    features_blob ={
    "table":table ,
    "id_cols":id_cols ,
    "x_cols":x_cols ,
    "y_cols":y_cols ,
    "notes":spec .get ("notes",{}),
    }
    (feature_dir /"features.json").write_text (json .dumps (features_blob ,indent =2 ),encoding ="utf-8")

    t1 =time .time ()
    print ("\n=== DONE ===")
    print (f"time={t1-t0:.2f}s")
    print (f"saved model:   {outdir / 'degradation_tcn.keras'}")
    print (f"saved scalers: {outdir / 'degradation_tcn.scalers.json'}")
    print (f"saved meta:    {outdir / 'degradation_tcn.meta.json'}")
    print (f"saved metrics: {outdir / 'degradation_tcn.metrics.json'}")
    print (f"saved features.json: {feature_dir / 'features.json'}")


if __name__ =="__main__":
    try :
        gpus =tf .config .list_physical_devices ("GPU")
        for g in gpus :
            tf .config .experimental .set_memory_growth (g ,True )
    except Exception :
        pass 

    main ()