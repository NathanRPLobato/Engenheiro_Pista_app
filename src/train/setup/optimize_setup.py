# arquivo: src/train/setup/optimize_setup.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
from pathlib import Path 
from typing import Dict ,List ,Tuple 

import numpy as np 
import pandas as pd 
import tensorflow as tf 


# -----------------------------
# helpers: carrega modelo + scalers
# -----------------------------
def load_artifacts (outdir :str ):
    outdir =Path (outdir )
    model =tf .keras .models .load_model (outdir /"setup_surrogate.keras")

    scalers =json .loads ((outdir /"setup_surrogate.scalers.json").read_text (encoding ="utf-8"))
    meta =json .loads ((outdir /"setup_surrogate.meta.json").read_text (encoding ="utf-8"))

    x_mean =np .array (scalers ["x_scaler"]["mean"],dtype =np .float32 )
    x_scale =np .array (scalers ["x_scaler"]["scale"],dtype =np .float32 )
    feature_cols =scalers ["x_scaler"]["feature_cols"]

    y_cols =scalers ["y_scaler"]["y_cols"]
    y_mean =np .array (scalers ["y_scaler"]["mean"],dtype =np .float32 )
    y_scale =np .array (scalers ["y_scaler"]["scale"],dtype =np .float32 )

    y_clip_lo =np .array (scalers ["y_scaler"]["clip_lo"],dtype =np .float32 )
    y_clip_hi =np .array (scalers ["y_scaler"]["clip_hi"],dtype =np .float32 )

    return {
    "model":model ,
    "feature_cols":feature_cols ,
    "x_mean":x_mean ,
    "x_scale":x_scale ,
    "y_cols":y_cols ,
    "y_mean":y_mean ,
    "y_scale":y_scale ,
    "y_clip_lo":y_clip_lo ,
    "y_clip_hi":y_clip_hi ,
    "meta":meta ,
    }


def x_scale_transform (X :np .ndarray ,x_mean :np .ndarray ,x_scale :np .ndarray )->np .ndarray :
    return (X -x_mean )/np .where (x_scale ==0 ,1.0 ,x_scale )


def y_inverse_transform (y_s :np .ndarray ,y_mean :np .ndarray ,y_scale :np .ndarray )->np .ndarray :
    return (y_s *np .where (y_scale ==0 ,1.0 ,y_scale ))+y_mean 


    # -----------------------------
    # Data -> realistic bounds
    # -----------------------------
def read_windows (sqlite_path :str ,table :str )->pd .DataFrame :
    con =sqlite3 .connect (sqlite_path )
    df =pd .read_sql (f"SELECT * FROM {table};",con )
    con .close ()
    return df 


def infer_setup_cols (df :pd .DataFrame )->List [str ]:
# tudo que termina com _setup OU é wing_setup_0/1 etc
    setup_like =[c for c in df .columns if c .endswith ("_setup")or "_setup_"in c or c .startswith ("wing_setup")]
    # inclui também tyre_press_setup_ etc
    setup_like =sorted (list (set (setup_like )))
    return setup_like 


def infer_bounds (df :pd .DataFrame ,setup_cols :List [str ],lo_p :float =2.0 ,hi_p :float =98.0 )->Dict [str ,Tuple [float ,float ]]:
    bounds ={}
    for c in setup_cols :
        x =pd .to_numeric (df [c ],errors ="coerce").dropna ()
        if len (x )<50 :
            continue 
        lo =float (np .percentile (x ,lo_p ))
        hi =float (np .percentile (x ,hi_p ))
        if lo ==hi :
            lo =float (x .min ())
            hi =float (x .max ())
        bounds [c ]=(lo ,hi )
    return bounds 


    # -----------------------------
    # Build one input row to the NN
    # -----------------------------
def make_base_row (df_track :pd .DataFrame ,feature_cols :List [str ],trackId :str )->Dict [str ,float ]:
    """
    Pegamos uma linha 'base' do dataset e depois substituímos só os parâmetros de setup.
    Isso evita faltar colunas e mantém distribuição realista do resto.
    """
    if df_track .empty :
        raise ValueError (f"Não achei dados para trackId={trackId} no dataset.")

        # pega uma linha mediana (robusto)
    base ={}
    for c in df_track .columns :
        if c in ["race_id"]:
            continue 
        if c =="trackId":
            base [c ]=str (trackId )
            continue 

        s =df_track [c ]
        if s .dtype =="O":
        # string/cat: pega moda
            v =s .dropna ().astype (str )
            base [c ]=v .mode ().iloc [0 ]if len (v )else ""
        else :
            v =pd .to_numeric (s ,errors ="coerce").dropna ()
            base [c ]=float (v .median ())if len (v )else 0.0 

            # agora transforma em features one-hot + num, respeitando feature_cols
            # Observação: durante treino você fez one-hot de trackId e colocou no feature_cols.
            # Aqui recriamos isso.
    row ={}
    for fc in feature_cols :
        row [fc ]=0.0 

        # preencher numéricos 'diretos' que existirem
    for k ,v in base .items ():
        if k in row and k !="trackId":
            row [k ]=float (v )if v is not None else 0.0 

            # one-hot trackId
            # fc exemplo: trackId_Montreal
    for fc in feature_cols :
        if fc .startswith ("trackId_"):
            row [fc ]=1.0 if fc ==f"trackId_{trackId}"else 0.0 

    return row 


    # -----------------------------
    # Objective
    # -----------------------------
def objective (y :Dict [str ,float ],style :str ="aggressive")->float :
    """
    Menor = melhor.
    Ajuste pesos conforme seu paper.

    aggressive: prioriza velocidade, aceita um pouco mais de stress.
    conservative: prioriza estabilidade + baixo stress (pneus/consistência).
    """
    under =abs (y ["understeer_index"])
    over =abs (y ["oversteer_index"])
    brake =abs (y ["brake_instability"])
    trct =abs (y ["traction_loss_exit"])
    corr =abs (y ["steering_correction_rate"])
    stress =abs (y ["stress_proxy"])
    speed =y ["speed__mean"]

    if style =="conservative":
    # estabilidade manda
        return (
        2.5 *under +
        2.5 *over +
        2.0 *brake +
        2.0 *trct +
        1.5 *corr +
        2.0 *stress +
        (-0.20 )*speed 
        )

        # aggressive (default)
    return (
    1.8 *under +
    1.8 *over +
    1.2 *brake +
    1.3 *trct +
    1.0 *corr +
    1.0 *stress +
    (-0.35 )*speed 
    )


    # -----------------------------
    # Optimizer: random + hillclimb
    # -----------------------------
def predict_from_row (art ,row :Dict [str ,float ])->Dict [str ,float ]:
    X =np .array ([[row [c ]for c in art ["feature_cols"]]],dtype =np .float32 )
    Xs =x_scale_transform (X ,art ["x_mean"],art ["x_scale"])
    y_s =art ["model"].predict (Xs ,verbose =0 ).astype (np .float32 )
    y =y_inverse_transform (y_s ,art ["y_mean"],art ["y_scale"])[0 ]

    # clip no mesmo range do treino (consistência)
    y =np .clip (y ,art ["y_clip_lo"],art ["y_clip_hi"])

    return {k :float (v )for k ,v in zip (art ["y_cols"],y )}


def sample_setup (bounds :Dict [str ,Tuple [float ,float ]],rng :np .random .Generator )->Dict [str ,float ]:
    s ={}
    for k ,(lo ,hi )in bounds .items ():
        s [k ]=float (rng .uniform (lo ,hi ))
    return s 


def apply_setup_to_row (row :Dict [str ,float ],setup :Dict [str ,float ])->Dict [str ,float ]:
    r =dict (row )
    for k ,v in setup .items ():
        if k in r :
            r [k ]=float (v )
    return r 


def optimize (
art ,
base_row :Dict [str ,float ],
bounds :Dict [str ,Tuple [float ,float ]],
style :str ,
n_random :int =2000 ,
n_steps :int =200 ,
step_frac :float =0.15 ,
seed :int =42 ,
):
    rng =np .random .default_rng (seed )

    # 1) random search
    best_setup =None 
    best_score =float ("inf")
    best_pred =None 

    for _ in range (n_random ):
        setup =sample_setup (bounds ,rng )
        row =apply_setup_to_row (base_row ,setup )
        pred =predict_from_row (art ,row )
        sc =objective (pred ,style =style )
        if sc <best_score :
            best_score =sc 
            best_setup =setup 
            best_pred =pred 

            # 2) hill-climb local
    keys =list (bounds .keys ())
    cur_setup =dict (best_setup )
    cur_score =best_score 
    cur_pred =best_pred 

    for _ in range (n_steps ):
        k =rng .choice (keys )
        lo ,hi =bounds [k ]
        span =hi -lo 
        if span <=0 :
            continue 

            # passo proporcional ao range, com ruído
        delta =rng .normal (0 ,step_frac *span )
        cand =dict (cur_setup )
        cand [k ]=float (np .clip (cand [k ]+delta ,lo ,hi ))

        row =apply_setup_to_row (base_row ,cand )
        pred =predict_from_row (art ,row )
        sc =objective (pred ,style =style )

        if sc <cur_score :
            cur_setup =cand 
            cur_score =sc 
            cur_pred =pred 

    return cur_setup ,cur_pred ,cur_score 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--windows_table",default ="setup_behavior_windows")
    ap .add_argument ("--model_dir",default ="models/champion/setup_surrogate_dl")
    ap .add_argument ("--trackId",required =True ,help ="Ex: Montreal, Losail, Las Vegas, Monaco (string igual no DB)")
    ap .add_argument ("--style",default ="aggressive",choices =["aggressive","conservative"])
    ap .add_argument ("--n_random",type =int ,default =2000 )
    ap .add_argument ("--n_steps",type =int ,default =250 )
    ap .add_argument ("--seed",type =int ,default =42 )
    args =ap .parse_args ()

    df =read_windows (args .sqlite_path ,args .windows_table )
    if df .empty :
        raise SystemExit ("Tabela de windows vazia.")

    df ["trackId"]=df ["trackId"].astype (str )

    # filtra pista
    df_track =df .loc [df ["trackId"]==str (args .trackId )].reset_index (drop =True )
    if df_track .empty :
        print ("tracks disponíveis:",sorted (df ["trackId"].astype (str ).unique ().tolist ()))
        raise SystemExit (f"Sem dados para trackId={args.trackId}")

    art =load_artifacts (args .model_dir )

    setup_cols =infer_setup_cols (df_track )
    bounds =infer_bounds (df_track ,setup_cols ,lo_p =2.0 ,hi_p =98.0 )
    if not bounds :
        raise SystemExit ("Não consegui inferir bounds de setup (colunas vazias/NaN).")

        # base row realista da pista
    base_row =make_base_row (df_track ,art ["feature_cols"],trackId =str (args .trackId ))

    best_setup ,best_pred ,best_score =optimize (
    art =art ,
    base_row =base_row ,
    bounds =bounds ,
    style =args .style ,
    n_random =args .n_random ,
    n_steps =args .n_steps ,
    seed =args .seed ,
    )

    print ("\n=== SETUP RECOMENDADO (surrogate-optimized) ===")
    print (f"trackId={args.trackId}  style={args.style}")
    print (f"score={best_score:.6f}\n")

    # imprime setup ordenado
    for k in sorted (best_setup .keys ()):
        print (f"{k:28s} = {best_setup[k]:.6f}")

    print ("\n=== PREDIÇÃO DO COMPORTAMENTO (targets) ===")
    for k in art ["y_cols"]:
        print (f"{k:24s} -> {best_pred.get(k, float('nan')):.6f}")


if __name__ =="__main__":
    main ()