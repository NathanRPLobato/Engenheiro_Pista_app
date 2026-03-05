from __future__ import annotations 

import argparse 
import json 
from pathlib import Path 

import numpy as np 
import pandas as pd 

from sklearn .model_selection import GroupShuffleSplit 
from sklearn .metrics import mean_absolute_error ,mean_squared_error ,r2_score 

import lightgbm as lgb 


def _load_features_json (feature_dir :Path )->dict :
    fp =feature_dir /"features.json"
    if not fp .exists ():
        raise SystemExit (f"features.json não encontrado em: {fp}")
    return json .loads (fp .read_text (encoding ="utf-8"))


def _load_index (feature_dir :Path ,index_name :str ="feature_index.parquet")->pd .DataFrame :
    idx_path =feature_dir /index_name 
    if not idx_path .exists ():
        raise SystemExit (
        f"Índice não encontrado: {idx_path}\n"
        f"Crie com: python -m src.feature_store.build_feature_index --feature_dir {feature_dir}"
        )
    idx =pd .read_parquet (idx_path )
    for c in ["trackId","race_id","path"]:
        if c not in idx .columns :
            raise SystemExit (f"Index inválido: faltando coluna {c}")
    idx ["trackId"]=idx ["trackId"].astype (str )
    idx ["race_id"]=idx ["race_id"].astype (str )
    return idx 


def _read_track_from_index (idx :pd .DataFrame ,trackId :str )->pd .DataFrame :
# lê só os parquets dessa pista
    sub =idx [idx ["trackId"]==str (trackId )].copy ()
    if sub .empty :
        return pd .DataFrame ()

    files =[Path (p )for p in sub ["path"].tolist ()if Path (p ).exists ()]
    if not files :
        return pd .DataFrame ()

    dfs =[pd .read_parquet (p )for p in files ]
    return pd .concat (dfs ,axis =0 ,ignore_index =True )


def _prep_xy (df :pd .DataFrame ,feature_cols :list [str ],categorical :list [str ])->tuple [pd .DataFrame ,np .ndarray ,pd .DataFrame ]:
    y =pd .to_numeric (df ["y_delta"],errors ="coerce").to_numpy ()
    ok =np .isfinite (y )
    df2 =df .loc [ok ].reset_index (drop =True )
    y2 =y [ok ]

    X =df2 [feature_cols ].copy ()

    for c in categorical :
        if c in X .columns :
            X [c ]=X [c ].astype ("category")

    for c in X .columns :
        if c in categorical :
            continue 
        if X [c ].dtype =="object":
            X [c ]=pd .to_numeric (X [c ],errors ="coerce")
        if pd .api .types .is_numeric_dtype (X [c ]):
            X [c ]=X [c ].astype (np .float32 )

    for c in X .columns :
        if c in categorical :
            X [c ]=X [c ].cat .add_categories (["__MISSING__"]).fillna ("__MISSING__")
        else :
            arr =X [c ].to_numpy ()
            med =np .nanmedian (arr )
            if not np .isfinite (med ):
                med =0.0 
            X [c ]=X [c ].fillna (med )

    return X ,y2 ,df2 


def _split_by_groups (
X :pd .DataFrame ,
y :np .ndarray ,
groups :np .ndarray ,
seed :int =42 ,
)->tuple [np .ndarray ,np .ndarray ,dict ]:
    uniq =np .unique (groups )
    n_groups =len (uniq )

    if n_groups >=3 :
        gss =GroupShuffleSplit (n_splits =1 ,test_size =0.20 ,random_state =seed )
        tr ,te =next (gss .split (X ,y ,groups =groups ))
        return tr ,te ,{"strategy":"GroupShuffleSplit","n_groups":n_groups ,"test_size":0.20 }

    if n_groups ==2 :
        rng =np .random .default_rng (seed )
        test_group =rng .choice (uniq ,size =1 )[0 ]
        te =np .where (groups ==test_group )[0 ]
        tr =np .where (groups !=test_group )[0 ]
        return tr ,te ,{"strategy":"HoldoutOneGroup","n_groups":n_groups ,"test_group":str (test_group )}

    tr =np .arange (len (y ))
    te =np .array ([],dtype =int )
    return tr ,te ,{"strategy":"NoValidationSingleGroup","n_groups":n_groups }


def _train_eval (
df_track :pd .DataFrame ,
feature_cols :list [str ],
categorical :list [str ],
device :str ,
gpu_platform_id :int ,
gpu_device_id :int ,
seed :int =42 ,
)->dict :
    if "race_id"not in df_track .columns :
        raise SystemExit ("Sem race_id no df_track.")

    groups =df_track ["race_id"].astype (str ).to_numpy ()

    X ,y ,df_clean =_prep_xy (df_track ,feature_cols ,categorical )
    tr ,te ,split_info =_split_by_groups (X ,y ,groups =df_clean ["race_id"].astype (str ).to_numpy (),seed =seed )

    metrics ={
    "split":split_info ,
    "n_rows":int (len (df_clean )),
    "n_features":int (X .shape [1 ]),
    "features_used":feature_cols ,
    "categorical":categorical ,
    "y_mean":float (np .mean (y ))if len (y )else None ,
    "y_std":float (np .std (y ))if len (y )else None ,
    }

    if len (te )>0 :
        yte =y [te ]
        pred0 =np .zeros_like (yte )
        metrics ["baseline_zero"]={
        "mae":float (mean_absolute_error (yte ,pred0 )),
        "rmse":float (np .sqrt (mean_squared_error (yte ,pred0 ))),
        "r2":float (r2_score (yte ,pred0 )),
        }
    else :
        metrics ["baseline_zero"]=None 

    params ={
    "objective":"regression_l1",
    "metric":["l1","rmse"],
    "learning_rate":0.03 ,
    "num_leaves":64 ,
    "min_data_in_leaf":15 ,
    "feature_fraction":0.9 ,
    "bagging_fraction":0.9 ,
    "bagging_freq":1 ,
    "lambda_l2":1.0 ,
    "verbosity":-1 ,
    "seed":seed ,
    }
    if device =="gpu":
        params .update ({
        "device":"gpu",
        "gpu_platform_id":gpu_platform_id ,
        "gpu_device_id":gpu_device_id ,
        })

    Xtr ,ytr =X .iloc [tr ],y [tr ]

    if len (te )==0 :
        dtrain =lgb .Dataset (Xtr ,label =ytr ,categorical_feature =categorical ,free_raw_data =False )
        model =lgb .train (params =params ,train_set =dtrain ,num_boost_round =400 )
        pred_tr =model .predict (Xtr )
        metrics ["lgbm"]={
        "mae":float (mean_absolute_error (ytr ,pred_tr )),
        "rmse":float (np .sqrt (mean_squared_error (ytr ,pred_tr ))),
        "r2":float (r2_score (ytr ,pred_tr )),
        "best_iter":int (model .current_iteration ()),
        "no_validation":True ,
        }
        return {"model":model ,"metrics":metrics }

    Xte ,yte =X .iloc [te ],y [te ]
    dtrain =lgb .Dataset (Xtr ,label =ytr ,categorical_feature =categorical ,free_raw_data =False )
    dvalid =lgb .Dataset (Xte ,label =yte ,categorical_feature =categorical ,free_raw_data =False )

    model =lgb .train (
    params =params ,
    train_set =dtrain ,
    valid_sets =[dvalid ],
    valid_names =["valid"],
    num_boost_round =4000 ,
    callbacks =[
    lgb .early_stopping (stopping_rounds =200 ,verbose =False ),
    lgb .log_evaluation (period =100 ),
    ],
    )

    pred =model .predict (Xte ,num_iteration =model .best_iteration )
    metrics ["lgbm"]={
    "mae":float (mean_absolute_error (yte ,pred )),
    "rmse":float (np .sqrt (mean_squared_error (yte ,pred ))),
    "r2":float (r2_score (yte ,pred )),
    "best_iter":int (model .best_iteration or 0 ),
    "no_validation":False ,
    }
    return {"model":model ,"metrics":metrics }


def main ()->None :
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--feature_dir",required =True )
    ap .add_argument ("--index_name",default ="feature_index.parquet")
    ap .add_argument ("--device",choices =["cpu","gpu"],default ="gpu")
    ap .add_argument ("--gpu_platform_id",type =int ,default =0 )
    ap .add_argument ("--gpu_device_id",type =int ,default =0 )
    ap .add_argument ("--out_dir",default ="models/champion/risk_pace_by_track")
    ap .add_argument ("--min_rows_track",type =int ,default =60 )
    args =ap .parse_args ()

    feature_dir =Path (args .feature_dir )
    out_root =Path (args .out_dir )
    out_root .mkdir (parents =True ,exist_ok =True )

    print ("\n=== TRAIN RISK_PACE LGBM BY TRACK (lazy by index) ===")
    print (f"feature_dir={feature_dir}")
    print (f"index_name={args.index_name}")
    print (f"device={args.device} gpu_platform_id={args.gpu_platform_id} gpu_device_id={args.gpu_device_id}")
    print (f"out_dir={out_root}")
    print (f"min_rows_track={args.min_rows_track}")

    cfg =_load_features_json (feature_dir )
    idx =_load_index (feature_dir ,args .index_name )

    cat =cfg .get ("categorical",[])
    all_feats =cfg .get ("all_features",[])
    ban =set (["race_ids_used"])
    # features só se existirem no parquet (checagem depois que ler df_t)
    all_feats =[c for c in all_feats if c not in ban ]

    base_feats =[c for c in all_feats if not c .startswith ("fp_")]
    fp_feats =all_feats [:]# inclui fp_

    results =[]
    tracks =sorted (idx ["trackId"].unique ().tolist ())

    for t in tracks :
        df_t =_read_track_from_index (idx ,t )
        if df_t .empty :
            continue 

        if "y_delta"not in df_t .columns :
            print (f"[skip] track={t} sem y_delta")
            continue 
        if "race_id"not in df_t .columns :
            print (f"[skip] track={t} sem race_id")
            continue 

        n_races =df_t ["race_id"].astype (str ).nunique ()
        if len (df_t )<args .min_rows_track :
            print (f"[skip] track={t} rows={len(df_t)} (min_rows_track={args.min_rows_track}) n_races={n_races}")
            continue 

            # filtra feature list p/ as colunas que existem mesmo nessa track (evita KeyError)
        base_feats_t =[c for c in base_feats if c in df_t .columns ]
        fp_feats_t =[c for c in fp_feats if c in df_t .columns ]

        yv =pd .to_numeric (df_t ["y_delta"],errors ="coerce")
        print (f"\n--- track={t} rows={len(df_t)} n_races={n_races} y_std={float(np.nanstd(yv)):.4f} y_mean={float(np.nanmean(yv)):.4f} ---")

        res_base =_train_eval (
        df_t ,base_feats_t ,cat ,
        device =args .device ,
        gpu_platform_id =args .gpu_platform_id ,
        gpu_device_id =args .gpu_device_id ,
        seed =42 ,
        )
        mae_base =res_base ["metrics"]["lgbm"]["mae"]
        print (f"[base] MAE={mae_base:.6f} RMSE={res_base['metrics']['lgbm']['rmse']:.6f} R2={res_base['metrics']['lgbm']['r2']:.4f} split={res_base['metrics']['split']['strategy']}")

        res_fp =_train_eval (
        df_t ,fp_feats_t ,cat ,
        device =args .device ,
        gpu_platform_id =args .gpu_platform_id ,
        gpu_device_id =args .gpu_device_id ,
        seed =42 ,
        )
        mae_fp =res_fp ["metrics"]["lgbm"]["mae"]
        print (f"[fp]   MAE={mae_fp:.6f} RMSE={res_fp['metrics']['lgbm']['rmse']:.6f} R2={res_fp['metrics']['lgbm']['r2']:.4f} split={res_fp['metrics']['split']['strategy']}")

        tag ,res =("fp",res_fp )if mae_fp <mae_base else ("base",res_base )
        print (f"[pick] track={t} -> {tag}")

        out_dir =out_root /str (t )
        out_dir .mkdir (parents =True ,exist_ok =True )

        model :lgb .Booster =res ["model"]
        metrics =res ["metrics"]

        model_path =out_dir /"lgbm.txt"
        model .save_model (str (model_path ))

        imp_gain =model .feature_importance (importance_type ="gain")
        imp_split =model .feature_importance (importance_type ="split")
        imp_df =pd .DataFrame ({
        "feature":model .feature_name (),
        "importance_gain":imp_gain ,
        "importance_split":imp_split ,
        }).sort_values ("importance_gain",ascending =False )
        imp_df .to_csv (out_dir /"feature_importance.csv",index =False )

        (out_dir /"metrics.json").write_text (json .dumps (metrics ,indent =2 ),encoding ="utf-8")

        results .append ({
        "trackId":t ,
        "rows":int (metrics ["n_rows"]),
        "n_races":int (n_races ),
        "picked":tag ,
        "mae":float (metrics ["lgbm"]["mae"]),
        "rmse":float (metrics ["lgbm"]["rmse"]),
        "r2":float (metrics ["lgbm"]["r2"]),
        "best_iter":int (metrics ["lgbm"]["best_iter"]),
        "n_features":int (metrics ["n_features"]),
        "split":metrics ["split"]["strategy"],
        "no_validation":bool (metrics ["lgbm"]["no_validation"]),
        })

    if not results :
        raise SystemExit ("Nenhuma pista treinada. Ajuste --min_rows_track.")

    summary =pd .DataFrame (results ).sort_values ("mae")
    print ("\n=== SUMMARY (best -> worst) ===")
    print (summary .to_string (index =False ))

    summary_path =out_root /"summary.csv"
    summary .to_csv (summary_path ,index =False )
    print (f"\n[ok] saved summary: {summary_path}")


if __name__ =="__main__":
    main ()