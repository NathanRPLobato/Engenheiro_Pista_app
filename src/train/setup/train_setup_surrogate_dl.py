# arquivo: src/train/setup/train_setup_surrogate_dl.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple 

import numpy as np 
import pandas as pd 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow .keras import layers 

from sklearn .model_selection import GroupShuffleSplit 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import mean_absolute_error ,mean_squared_error ,r2_score 


@dataclass 
class Cfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"
    table :str ="setup_behavior_windows"
    outdir :str ="models/champion/setup_surrogate_dl"

    test_size :float =0.20 
    seed :int =42 

    epochs :int =400 
    batch_size :int =512 
    lr :float =1e-3 
    patience :int =60 

    hidden1 :int =256 
    hidden2 :int =128 
    hidden3 :int =64 
    dropout :float =0.15 
    l2 :float =1e-6 

    min_samples_window :int =40 

    # >>> NOVO: robustez no Y
    y_clip_lo :float =1.0 # percentil
    y_clip_hi :float =99.0 # percentil
    scale_y :bool =True 


def _ensure_dir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _read_sqlite (cfg :Cfg )->pd .DataFrame :
    con =sqlite3 .connect (cfg .sqlite_path )
    df =pd .read_sql (f"SELECT * FROM {cfg.table};",con )
    con .close ()
    return df 


def _targets ()->List [str ]:
    return [
    "understeer_index",
    "oversteer_index",
    "brake_instability",
    "traction_loss_exit",
    "steering_correction_rate",
    "stress_proxy",
    "speed__mean",
    ]


def _pick_columns (df :pd .DataFrame )->Tuple [List [str ],List [str ],List [str ]]:
    id_cols =["race_id","trackId","lap_number","d0","d1","n_samples","t_start","t_end"]
    cat_cols =["trackId"]
    ignore =set (id_cols +cat_cols )
    numeric_cols =[c for c in df .columns if c not in ignore ]
    return id_cols ,cat_cols ,numeric_cols 


def _clean (df :pd .DataFrame ,cfg :Cfg )->pd .DataFrame :
    df =df .copy ()

    df ["race_id"]=df ["race_id"].astype (str )
    df ["trackId"]=df ["trackId"].astype (str )

    if "n_samples"in df .columns :
        df ["n_samples"]=pd .to_numeric (df ["n_samples"],errors ="coerce")
        df =df .loc [df ["n_samples"].fillna (0 )>=cfg .min_samples_window ].reset_index (drop =True )

    for y in _targets ():
        if y in df .columns :
            df [y ]=pd .to_numeric (df [y ],errors ="coerce")

    ys =[y for y in _targets ()if y in df .columns ]
    df =df .dropna (subset =ys ).reset_index (drop =True )

    return df 


def _build_model (n_in :int ,n_out :int ,cfg :Cfg )->keras .Model :
    inp =keras .Input (shape =(n_in ,),name ="x")

    x =layers .Dense (cfg .hidden1 ,activation ="relu",kernel_regularizer =keras .regularizers .l2 (cfg .l2 ))(inp )
    x =layers .Dropout (cfg .dropout )(x )

    x =layers .Dense (cfg .hidden2 ,activation ="relu",kernel_regularizer =keras .regularizers .l2 (cfg .l2 ))(x )
    x =layers .Dropout (cfg .dropout )(x )

    x =layers .Dense (cfg .hidden3 ,activation ="relu",kernel_regularizer =keras .regularizers .l2 (cfg .l2 ))(x )

    out =layers .Dense (n_out ,activation ="linear",name ="y")(x )

    model =keras .Model (inp ,out )
    model .compile (
    optimizer =keras .optimizers .Adam (learning_rate =cfg .lr ),
    loss ="huber",
    metrics =[keras .metrics .MeanAbsoluteError (name ="mae")],
    )
    return model 


def _eval_per_target (y_true :np .ndarray ,y_pred :np .ndarray ,y_cols :List [str ])->Dict :
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


def _make_Xy_train (df_tr :pd .DataFrame ,cat_cols :List [str ],num_cols :List [str ],y_cols :List [str ]):
    Xn =df_tr [num_cols ].copy ()
    for c in num_cols :
        Xn [c ]=pd .to_numeric (Xn [c ],errors ="coerce")
    num_medians =Xn .median (numeric_only =True )
    Xn =Xn .fillna (num_medians )

    Xc_parts =[]
    cat_onehot ={}
    for c in cat_cols :
        oh =pd .get_dummies (df_tr [c ].astype (str ),prefix =c )
        Xc_parts .append (oh )
        cat_onehot [c ]=oh .columns .tolist ()
    Xc =pd .concat (Xc_parts ,axis =1 )if Xc_parts else pd .DataFrame (index =df_tr .index )

    Xtr_df =pd .concat ([Xn ,Xc ],axis =1 )
    ytr =df_tr [y_cols ].to_numpy (dtype =np .float32 )

    meta ={
    "num_cols":num_cols ,
    "cat_cols":cat_cols ,
    "cat_onehot":cat_onehot ,
    "feature_cols":Xtr_df .columns .tolist (),
    "num_medians":{k :float (v )for k ,v in num_medians .items ()},
    "y_cols":y_cols ,
    }
    return Xtr_df ,ytr ,meta 


def _make_X_test (df_te :pd .DataFrame ,meta :Dict )->pd .DataFrame :
    num_cols =meta ["num_cols"]
    cat_cols =meta ["cat_cols"]
    feature_cols =meta ["feature_cols"]
    num_medians =meta ["num_medians"]
    cat_onehot =meta ["cat_onehot"]

    Xn =df_te [num_cols ].copy ()
    for c in num_cols :
        Xn [c ]=pd .to_numeric (Xn [c ],errors ="coerce")
        Xn [c ]=Xn [c ].fillna (num_medians .get (c ,float (pd .to_numeric (Xn [c ],errors ="coerce").median ())))

    Xc_parts =[]
    for c in cat_cols :
        oh =pd .get_dummies (df_te [c ].astype (str ),prefix =c )
        expected =cat_onehot .get (c ,[])
        oh =oh .reindex (columns =expected ,fill_value =0 )
        Xc_parts .append (oh )
    Xc =pd .concat (Xc_parts ,axis =1 )if Xc_parts else pd .DataFrame (index =df_te .index )

    Xte_df =pd .concat ([Xn ,Xc ],axis =1 )
    Xte_df =Xte_df .reindex (columns =feature_cols ,fill_value =0 )
    return Xte_df 


def _clip_y (y :np .ndarray ,lo_p :float ,hi_p :float )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """
    clip por coluna usando percentis do TREINO
    """
    lo =np .percentile (y ,lo_p ,axis =0 )
    hi =np .percentile (y ,hi_p ,axis =0 )
    y2 =np .clip (y ,lo ,hi )
    return y2 ,lo ,hi 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--table",default ="setup_behavior_windows")
    ap .add_argument ("--outdir",default ="models/champion/setup_surrogate_dl")
    ap .add_argument ("--epochs",type =int ,default =400 )
    ap .add_argument ("--batch_size",type =int ,default =512 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--test_size",type =float ,default =0.20 )
    ap .add_argument ("--seed",type =int ,default =42 )
    ap .add_argument ("--min_samples_window",type =int ,default =40 )
    ap .add_argument ("--y_clip_lo",type =float ,default =1.0 )
    ap .add_argument ("--y_clip_hi",type =float ,default =99.0 )
    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    table =args .table ,
    outdir =args .outdir ,
    epochs =args .epochs ,
    batch_size =args .batch_size ,
    lr =args .lr ,
    test_size =args .test_size ,
    seed =args .seed ,
    min_samples_window =args .min_samples_window ,
    y_clip_lo =args .y_clip_lo ,
    y_clip_hi =args .y_clip_hi ,
    )

    outdir =Path (cfg .outdir )
    _ensure_dir (outdir )

    print ("\n=== TRAIN SETUP SURROGATE (DL multi-output) ===")
    print (f"sqlite={cfg.sqlite_path}")
    print (f"table={cfg.table}")
    print (f"outdir={cfg.outdir}")

    t0 =time .time ()
    df =_read_sqlite (cfg )
    if df .empty :
        raise SystemExit ("Tabela vazia.")

    df =_clean (df ,cfg )

    y_cols =[c for c in _targets ()if c in df .columns ]
    if not y_cols :
        raise SystemExit (f"Não achei targets {_targets()} na tabela.")

    id_cols ,cat_cols ,numeric_candidates =_pick_columns (df )
    num_cols =[c for c in numeric_candidates if c not in set (id_cols +cat_cols +y_cols )]

    groups =df ["race_id"].astype (str ).to_numpy ()
    gss =GroupShuffleSplit (n_splits =1 ,test_size =cfg .test_size ,random_state =cfg .seed )
    tr_idx ,te_idx =next (gss .split (df ,groups =groups ))

    df_tr =df .iloc [tr_idx ].reset_index (drop =True )
    df_te =df .iloc [te_idx ].reset_index (drop =True )

    Xtr_df ,ytr_raw ,meta =_make_Xy_train (df_tr ,cat_cols ,num_cols ,y_cols )
    Xte_df =_make_X_test (df_te ,meta )
    yte_raw =df_te [y_cols ].to_numpy (dtype =np .float32 )

    # >>> CLIP Y com base no TREINO
    ytr_clip ,y_lo ,y_hi =_clip_y (ytr_raw ,cfg .y_clip_lo ,cfg .y_clip_hi )
    yte_clip =np .clip (yte_raw ,y_lo ,y_hi )

    # >>> SCALE X
    x_scaler =StandardScaler ()
    Xtr_s =x_scaler .fit_transform (Xtr_df .to_numpy (dtype =np .float32 ))
    Xte_s =x_scaler .transform (Xte_df .to_numpy (dtype =np .float32 ))

    # >>> SCALE Y (treina no espaço normalizado)
    y_scaler =StandardScaler ()
    ytr =y_scaler .fit_transform (ytr_clip )
    yte =y_scaler .transform (yte_clip )

    model =_build_model (n_in =Xtr_s .shape [1 ],n_out =ytr .shape [1 ],cfg =cfg )

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

    # predict no espaço escalado e volta pro espaço real
    ypred_s =model .predict (Xte_s ,verbose =0 )
    ypred =y_scaler .inverse_transform (ypred_s )

    metrics =_eval_per_target (yte_clip ,ypred ,y_cols )

    # prints por target (pra você ver o assassino do RMSE)
    print ("\n=== METRICS PER TARGET (real space, clipped) ===")
    for k in y_cols :
        print (f"{k:24s} MAE={metrics[k]['mae']:.4f} RMSE={metrics[k]['rmse']:.4f} R2={metrics[k]['r2']:.4f}")
    print ("\nMetrics (macro):",metrics ["macro"])

    # save
    (outdir /"setup_surrogate.keras").write_bytes (b"")# placeholder p/ Windows locks raros
    model .save (outdir /"setup_surrogate.keras")

    blob ={
    "x_scaler":{
    "mean":x_scaler .mean_ .tolist (),
    "scale":x_scaler .scale_ .tolist (),
    "var":x_scaler .var_ .tolist (),
    "n_features":int (len (x_scaler .mean_ )),
    "feature_cols":meta ["feature_cols"],
    },
    "y_scaler":{
    "mean":y_scaler .mean_ .tolist (),
    "scale":y_scaler .scale_ .tolist (),
    "var":y_scaler .var_ .tolist (),
    "y_cols":y_cols ,
    "clip_lo_p":cfg .y_clip_lo ,
    "clip_hi_p":cfg .y_clip_hi ,
    "clip_lo":y_lo .tolist (),
    "clip_hi":y_hi .tolist (),
    },
    }
    (outdir /"setup_surrogate.scalers.json").write_text (json .dumps (blob ,indent =2 ),encoding ="utf-8")
    (outdir /"setup_surrogate.meta.json").write_text (json .dumps (meta ,indent =2 ),encoding ="utf-8")
    (outdir /"setup_surrogate.metrics.json").write_text (json .dumps (metrics ,indent =2 ),encoding ="utf-8")

    t1 =time .time ()
    print ("\n=== DONE ===")
    print (f"rows={len(df)}  X={Xtr_s.shape[1]}  Y={len(y_cols)}  time={t1-t0:.2f}s")
    print (f"saved: {outdir / 'setup_surrogate.keras'}")
    print (f"saved: {outdir / 'setup_surrogate.scalers.json'}")


if __name__ =="__main__":
    try :
        gpus =tf .config .list_physical_devices ("GPU")
        for g in gpus :
            tf .config .experimental .set_memory_growth (g ,True )
    except Exception :
        pass 

    main ()