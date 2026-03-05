# arquivo: src/train/fingerprint/autoencoder_fingerprint.py
# Autoencoder (tabular) p/ aprender embedding de 'driver fingerprint' a partir de janel…
#
# Ideia:
# - Entrada: features agregadas por janela (mean/max/min etc) derivadas de inputs do pi…
# - Saída: reconstrução da própria entrada.
# - Embedding: vetor latente (fingerprint_vector) que representa o estilo.
#
# Uso recomendado:
# 1) Garanta que você tem uma tabela com janelas agregadas, ex: telemetry_windows
#
# 2) Rode este script p/ treinar o autoencoder e gerar embeddings por janela.
#
# Depois:
# - Você agrega embeddings por corrida/pista/pneu (média/mediana) p/ obter o fingerprin…
# - Injeta o fingerprint no modelo Risk Pace como features adicionais.

from __future__ import annotations 

import argparse 
import json 
import os 
import sqlite3 
from dataclasses import dataclass 
from pathlib import Path 
from typing import List ,Tuple 

import numpy as np 
import pandas as pd 

# sklearn
from sklearn .model_selection import GroupShuffleSplit 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import mean_squared_error 

# tensorflow
import tensorflow as tf 
from tensorflow import keras 
from tensorflow .keras import layers 


DEFAULT_SQLITE =Path ("data/refined/telemetry.sqlite")
DEFAULT_TABLE ="telemetry_windows"# ajuste se seu nome for outro
DEFAULT_OUTDIR =Path ("models/champion/fingerprint_ae")


@dataclass 
class AEConfig :
    latent_dim :int =16 
    hidden_1 :int =128 
    hidden_2 :int =64 
    dropout :float =0.10 
    lr :float =1e-3 
    batch_size :int =256 
    epochs :int =200 
    early_stop_patience :int =20 
    random_state :int =42 


def _set_seed (seed :int )->None :
    os .environ ["PYTHONHASHSEED"]=str (seed )
    np .random .seed (seed )
    tf .random .set_seed (seed )


def _read_windows (sqlite_path :Path ,table :str )->pd .DataFrame :
    con =sqlite3 .connect (sqlite_path )
    try :
        df =pd .read_sql (f"SELECT * FROM {table};",con )
    finally :
        con .close ()
    return df 


def _infer_feature_columns (df :pd .DataFrame )->List [str ]:
    """
    Seleciona automaticamente colunas numéricas úteis para fingerprint.
    Regras:
    - Remove identificadores e alvos óbvios (race_id, t0, t1, y_*, baseline_*, lap_* etc).
    - Mantém features com sufixos agregados (__mean/__max/__min/__std etc) e algumas colunas contínuas comuns.
    """
    block_prefix =(
    "race_id","trackId","lap_number","compound_id",
    "t0","t1","n_samples",
    "lap_time","dt_window",
    "baseline_","y_","lap_end_"
    )

    # colunas que são 'meta' e não devem virar input do AE
    block_exact ={
    "race_ids_used",# quando existir
    }

    cols =[]
    for c in df .columns :
        if c in block_exact :
            continue 

        if any (c .startswith (p )for p in block_prefix ):
            continue 

            # mantém colunas agregadas e algumas contínuas típicas
        if "__"in c :
            cols .append (c )
            continue 

            # fallback: colunas contínuas sem sufixo (caso você tenha)
        if c in {"fuel","speed","rpm","gear","gforce_X","gforce_Y","gforce_Z"}:
            cols .append (c )

            # filtra p/ numéricas ou convertíveis
    out =[]
    for c in cols :
        s =pd .to_numeric (df [c ],errors ="coerce")
        if np .isfinite (s ).sum ()>0 :
            out .append (c )

    return sorted (set (out ))


def _clean_numeric_matrix (df :pd .DataFrame ,feature_cols :List [str ])->Tuple [pd .DataFrame ,np .ndarray ]:
    """
    Converte colunas para numérico, remove linhas com muitos NaNs e preenche NaN com mediana.
    """
    X_df =df [feature_cols ].copy ()

    for c in feature_cols :
        X_df [c ]=pd .to_numeric (X_df [c ],errors ="coerce")

        # remove linhas com NaN demais (ex: 50%+ faltando)
    nan_ratio =X_df .isna ().mean (axis =1 )
    keep =nan_ratio <0.50 
    X_df =X_df .loc [keep ].reset_index (drop =True )
    df_kept =df .loc [keep ].reset_index (drop =True )

    # preenche NaN com mediana da coluna
    med =X_df .median (numeric_only =True )
    X_df =X_df .fillna (med )

    X =X_df .values .astype (np .float32 )
    return df_kept ,X 


def build_autoencoder (input_dim :int ,cfg :AEConfig )->Tuple [keras .Model ,keras .Model ]:
    """
    Retorna:
    - autoencoder: modelo completo (entrada -> reconstrução)
    - encoder: modelo (entrada -> embedding)
    """
    inp =keras .Input (shape =(input_dim ,),name ="x")

    x =layers .Dense (cfg .hidden_1 ,activation ="relu")(inp )
    x =layers .Dropout (cfg .dropout )(x )
    x =layers .Dense (cfg .hidden_2 ,activation ="relu")(x )
    x =layers .Dropout (cfg .dropout )(x )

    z =layers .Dense (cfg .latent_dim ,activation =None ,name ="fingerprint_vector")(x )

    x =layers .Dense (cfg .hidden_2 ,activation ="relu")(z )
    x =layers .Dropout (cfg .dropout )(x )
    x =layers .Dense (cfg .hidden_1 ,activation ="relu")(x )
    out =layers .Dense (input_dim ,activation =None ,name ="x_hat")(x )

    autoencoder =keras .Model (inp ,out ,name ="fingerprint_autoencoder")
    encoder =keras .Model (inp ,z ,name ="fingerprint_encoder")

    opt =keras .optimizers .Adam (learning_rate =cfg .lr )
    autoencoder .compile (optimizer =opt ,loss ="mse")
    return autoencoder ,encoder 


def main ()->None :
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (DEFAULT_SQLITE ))
    ap .add_argument ("--table",default =DEFAULT_TABLE )
    ap .add_argument ("--outdir",default =str (DEFAULT_OUTDIR ))
    ap .add_argument ("--latent_dim",type =int ,default =16 )
    ap .add_argument ("--hidden_1",type =int ,default =128 )
    ap .add_argument ("--hidden_2",type =int ,default =64 )
    ap .add_argument ("--dropout",type =float ,default =0.10 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--batch_size",type =int ,default =256 )
    ap .add_argument ("--epochs",type =int ,default =200 )
    ap .add_argument ("--patience",type =int ,default =20 )
    ap .add_argument ("--device",choices =["auto","cpu","gpu"],default ="auto")
    ap .add_argument ("--test_size",type =float ,default =0.20 )
    ap .add_argument ("--group_col",default ="race_id",help ="coluna para split por grupo (evita vazamento)")
    ap .add_argument ("--feature_json_only",action ="store_true",help ="só imprime e salva a lista de features detectadas")
    args =ap .parse_args ()

    sqlite_path =Path (args .sqlite_path )
    outdir =Path (args .outdir )
    outdir .mkdir (parents =True ,exist_ok =True )

    cfg =AEConfig (
    latent_dim =args .latent_dim ,
    hidden_1 =args .hidden_1 ,
    hidden_2 =args .hidden_2 ,
    dropout =args .dropout ,
    lr =args .lr ,
    batch_size =args .batch_size ,
    epochs =args .epochs ,
    early_stop_patience =args .patience ,
    random_state =42 ,
    )
    _set_seed (cfg .random_state )

    # controle simples de device
    if args .device =="cpu":
        os .environ ["CUDA_VISIBLE_DEVICES"]="-1"
    elif args .device =="gpu":
    # deixa o TF pegar GPU se existir
        pass 

    print ("\n=== TRAIN FINGERPRINT AUTOENCODER (TABULAR) ===")
    print (f"sqlite={sqlite_path}")
    print (f"table={args.table}")
    print (f"outdir={outdir}")
    print (f"cfg: latent_dim={cfg.latent_dim} hidden_1={cfg.hidden_1} hidden_2={cfg.hidden_2} dropout={cfg.dropout} lr={cfg.lr}")

    df =_read_windows (sqlite_path ,args .table )
    if df .empty :
        raise SystemExit ("Tabela de janelas está vazia. Primeiro gere as windows.")

        # detecta features
    feature_cols =_infer_feature_columns (df )
    if len (feature_cols )<8 :
        raise SystemExit (
        f"Poucas features detectadas ({len(feature_cols)}). "
        "Verifique se sua tabela tem colunas agregadas tipo throttle__mean etc."
        )

        # salva lista de features
    features_path =outdir /"fingerprint_ae.features.json"
    features_path .write_text (json .dumps (feature_cols ,indent =2 ,ensure_ascii =False ),encoding ="utf-8")
    print (f"[ok] features detectadas: {len(feature_cols)}  salvo em: {features_path}")

    if args .feature_json_only :
        print ("\nfeatures:")
        for c in feature_cols [:40 ]:
            print (" -",c )
        if len (feature_cols )>40 :
            print (" ...")
        return 

        # monta X numérico e limpa
    df_kept ,X =_clean_numeric_matrix (df ,feature_cols )
    print (f"[ok] rows_before={len(df)} rows_after_clean={len(df_kept)} input_dim={X.shape[1]}")

    # split por grupo p/ não vazar corrida
    if args .group_col not in df_kept .columns :
    # fallback: split aleatório se não tiver race_id
        idx =np .arange (len (df_kept ))
        np .random .shuffle (idx )
        cut =int ((1.0 -args .test_size )*len (idx ))
        tr_idx ,te_idx =idx [:cut ],idx [cut :]
    else :
        groups =df_kept [args .group_col ].astype (str ).values 
        gss =GroupShuffleSplit (n_splits =1 ,test_size =args .test_size ,random_state =cfg .random_state )
        tr_idx ,te_idx =next (gss .split (X ,X ,groups =groups ))

    X_tr ,X_te =X [tr_idx ],X [te_idx ]

    # normalização
    scaler =StandardScaler ()
    X_tr_s =scaler .fit_transform (X_tr ).astype (np .float32 )
    X_te_s =scaler .transform (X_te ).astype (np .float32 )

    # build modelo
    autoencoder ,encoder =build_autoencoder (input_dim =X_tr_s .shape [1 ],cfg =cfg )
    autoencoder .summary ()

    callbacks =[
    keras .callbacks .EarlyStopping (
    monitor ="val_loss",
    patience =cfg .early_stop_patience ,
    restore_best_weights =True ,
    ),
    keras .callbacks .ReduceLROnPlateau (
    monitor ="val_loss",
    factor =0.5 ,
    patience =max (5 ,cfg .early_stop_patience //3 ),
    min_lr =1e-6 ,
    verbose =1 ,
    ),
    ]

    hist =autoencoder .fit (
    X_tr_s ,X_tr_s ,
    validation_data =(X_te_s ,X_te_s ),
    epochs =cfg .epochs ,
    batch_size =cfg .batch_size ,
    verbose =1 ,
    callbacks =callbacks ,
    shuffle =True ,
    )

    # métricas simples
    Xhat_te =autoencoder .predict (X_te_s ,batch_size =cfg .batch_size ,verbose =0 )
    mse_te =mean_squared_error (X_te_s .reshape (-1 ),Xhat_te .reshape (-1 ))
    rmse_te =float (np .sqrt (mse_te ))
    print (f"\n[ok] reconstruction RMSE (scaled space) = {rmse_te:.6f}")

    # salva modelos
    ae_path =outdir /"fingerprint_autoencoder.keras"
    enc_path =outdir /"fingerprint_encoder.keras"
    autoencoder .save (ae_path )
    encoder .save (enc_path )
    print (f"[ok] saved: {ae_path}")
    print (f"[ok] saved: {enc_path}")

    # salva scaler
    # sem joblib p/ evitar dependência extra: salva mean/scale em json
    scaler_path =outdir /"fingerprint_scaler.json"
    scaler_blob ={
    "mean":scaler .mean_ .tolist (),
    "scale":scaler .scale_ .tolist (),
    "var":scaler .var_ .tolist (),
    "n_features":int (len (feature_cols )),
    "features":feature_cols ,
    }
    scaler_path .write_text (json .dumps (scaler_blob ,indent =2 ,ensure_ascii =False ),encoding ="utf-8")
    print (f"[ok] saved scaler: {scaler_path}")

    # gera embeddings p/ TODAS as janelas e salva parquet
    X_all_s =scaler .transform (X ).astype (np .float32 )
    Z =encoder .predict (X_all_s ,batch_size =cfg .batch_size ,verbose =0 )
    z_cols =[f"fp_{i:02d}"for i in range (Z .shape [1 ])]

    out_df =df_kept .copy ()

    # preserva algumas colunas chaves se existirem
    keep_meta =[c for c in ["race_id","trackId","lap_number","compound_id","t0","t1","n_samples"]if c in out_df .columns ]
    out_df =out_df [keep_meta ].copy ()

    for i ,c in enumerate (z_cols ):
        out_df [c ]=Z [:,i ].astype (np .float32 )

    emb_path =outdir /"fingerprint_embeddings_windows.parquet"
    out_df .to_parquet (emb_path ,index =False )
    print (f"[ok] embeddings por janela salvos em: {emb_path}")

    # também salva uma versão agregada por corrida (média)
    if "race_id"in out_df .columns :
        grp =out_df .groupby ("race_id",as_index =False )[z_cols ].mean (numeric_only =True )
        race_path =outdir /"fingerprint_embeddings_by_race.parquet"
        grp .to_parquet (race_path ,index =False )
        print (f"[ok] embeddings por corrida salvos em: {race_path}")

        # salva métricas/treino
    metrics ={
    "recon_rmse_scaled":rmse_te ,
    "rows_total":int (len (df )),
    "rows_used":int (len (df_kept )),
    "input_dim":int (X .shape [1 ]),
    "latent_dim":int (cfg .latent_dim ),
    "epochs_trained":int (len (hist .history .get ("loss",[]))),
    }
    metrics_path =outdir /"fingerprint_ae.metrics.json"
    metrics_path .write_text (json .dumps (metrics ,indent =2 ,ensure_ascii =False ),encoding ="utf-8")
    print (f"[ok] metrics: {metrics_path}")

    print ("\nOK. Próximo passo: usar fp_00..fp_15 como features no Risk Pace e comparar ganho.")


if __name__ =="__main__":
    main ()
