# arquivo: src/train/setup/generate_recommended_setups.py
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


# -----------------------------
# Config
# -----------------------------
@dataclass 
class Cfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"
    table :str ="setup_behavior_windows"
    model_dir :str ="models/champion/setup_surrogate_dl"

    out_json :str ="data/recommended_setups/setup_recommendations.json"
    out_table :str ="recommended_setups"
    drop_and_rebuild :bool =False 

    n_iter :int =6000 
    seed :int =42 

    # modos de otimização (3 modos)
    modes :Tuple [str ,...]=("aggressive","balanced","safe")

    # tamanhos e elite do CEM
    batch_size :int =1024 
    elite_frac :float =0.10 
    n_rounds :int =6 

    # clipping do sampling
    clip_sigma_min :float =0.05 
    clip_sigma_max :float =2.50 

    # pit (não usado aqui, mas deixo guardado pra simulação)
    pit_loss_sec :float =22.0 


    # -----------------------------
    # Utils I/O
    # -----------------------------
def _ensure_dir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _read_json (path :Path )->Dict :
    return json .loads (path .read_text (encoding ="utf-8"))


def _write_json (path :Path ,obj :Dict )->None :
    path .write_text (json .dumps (obj ,indent =2 ,ensure_ascii =False ),encoding ="utf-8")


def _connect (db_path :str )->sqlite3 .Connection :
    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    return con 


def _drop_table (con :sqlite3 .Connection ,name :str )->None :
    con .execute (f"DROP TABLE IF EXISTS {name};")


def _create_table (con :sqlite3 .Connection ,name :str )->None :
    con .execute (f"""
        CREATE TABLE IF NOT EXISTS {name} (
            trackId TEXT,
            mode TEXT,
            score REAL,
            n_candidates INT,
            model_dim INT,
            y_pred_json TEXT,
            setup_full_json TEXT,
            x_full_json TEXT,
            x_context_json TEXT,
            created_at TEXT
        );
    """)


def _insert_row (con :sqlite3 .Connection ,name :str ,row :Dict )->None :
    con .execute (
    f"""
        INSERT INTO {name}
        (trackId, mode, score, n_candidates, model_dim, y_pred_json, setup_full_json, x_full_json, x_context_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'));
        """,
    (
    row ["trackId"],
    row ["mode"],
    float (row ["score"]),
    int (row ["n_candidates"]),
    int (row ["model_dim"]),
    json .dumps (row ["y_pred"],ensure_ascii =False ),
    json .dumps (row ["setup_full"],ensure_ascii =False ),
    json .dumps (row ["x_full"],ensure_ascii =False ),
    json .dumps (row ["x_context"],ensure_ascii =False ),
    )
    )


    # -----------------------------
    # Feature helpers
    # -----------------------------
def _is_setup_col (c :str )->bool :
# parâmetros fixos de setup
    if c .endswith ("_setup"):
        return True 
    if c .startswith ("wing_setup_"):
        return True 
    if c .startswith ("camber_setup_")or c .startswith ("toe_setup_"):
        return True 
    if c .startswith ("susp_spring_setup_")or c .startswith ("susp_height_setup_"):
        return True 
    if c .startswith ("arb_setup_"):
        return True 
    if c .startswith ("tyre_press_setup_"):
        return True 
    if c in ("front_brake_bias",):
        return True 
    return False 


def _split_x_into_setup_and_context (x_dict :Dict )->Tuple [Dict ,Dict ]:
    setup ={}
    context ={}
    for k ,v in x_dict .items ():
        (setup if _is_setup_col (k )else context )[k ]=v 
    return setup ,context 


def _sanitize_float (v ):
# garante JSON limpo, sem numpy types
    if v is None :
        return None 
    try :
        if hasattr (v ,"item"):
            v =v .item ()
    except Exception :
        pass 
    if isinstance (v ,(np .floating ,float )):
        if np .isnan (v )or np .isinf (v ):
            return None 
        return float (v )
    if isinstance (v ,(np .integer ,int )):
        return int (v )
    if isinstance (v ,(str ,)):
        return v 
    return v 


    # -----------------------------
    # carrega modelo + scalers
    # -----------------------------
def _load_model (model_dir :Path )->keras .Model :
    path =model_dir /"setup_surrogate.keras"
    if not path .exists ():
        raise FileNotFoundError (f"Não achei modelo: {path}")
    return keras .models .load_model (path )


def _load_meta (model_dir :Path )->Dict :
    path =model_dir /"setup_surrogate.meta.json"
    if not path .exists ():
        raise FileNotFoundError (f"Não achei meta: {path}")
    meta =_read_json (path )
    if "feature_cols"not in meta :
        raise ValueError ("meta.json sem 'feature_cols'")
    return meta 


def _load_scalers (model_dir :Path )->Dict :
    path =model_dir /"setup_surrogate.scalers.json"
    if not path .exists ():
        raise FileNotFoundError (f"Não achei scalers: {path}")
    j =_read_json (path )
    if "x_scaler"not in j or "y_scaler"not in j :
        raise ValueError ("scalers.json precisa ter x_scaler e y_scaler")
    if "mean"not in j ["x_scaler"]or "scale"not in j ["x_scaler"]:
        raise ValueError ("x_scaler sem mean/scale")
    if "mean"not in j ["y_scaler"]or "scale"not in j ["y_scaler"]:
        raise ValueError ("y_scaler sem mean/scale")
    return j 


def _x_transform (X :np .ndarray ,x_scaler :Dict )->np .ndarray :
    mean =np .asarray (x_scaler ["mean"],dtype =np .float32 )
    scale =np .asarray (x_scaler ["scale"],dtype =np .float32 )
    if X .shape [1 ]!=mean .shape [0 ]:
        raise ValueError (f"X dim {X.shape[1]} != scaler dim {mean.shape[0]}")
    return (X -mean )/(scale +1e-12 )


def _y_inverse_transform (Ys :np .ndarray ,y_scaler :Dict )->np .ndarray :
    mean =np .asarray (y_scaler ["mean"],dtype =np .float32 )
    scale =np .asarray (y_scaler ["scale"],dtype =np .float32 )
    if Ys .shape [1 ]!=mean .shape [0 ]:
        raise ValueError (f"Y dim {Ys.shape[1]} != scaler dim {mean.shape[0]}")
    return Ys *(scale +1e-12 )+mean 


    # -----------------------------
    # Read dataset for ranges + tracks
    # -----------------------------
def _read_df (con :sqlite3 .Connection ,table :str )->pd .DataFrame :
    df =pd .read_sql (f"SELECT * FROM {table};",con )
    if df .empty :
        raise ValueError (f"Tabela {table} está vazia.")
        # trackId aqui é TEXT (no seu build_setup_behavior_windows)
    if "trackId"not in df .columns :
        raise ValueError ("Tabela sem trackId")
    df ["trackId"]=df ["trackId"].astype (str )
    return df 


def _get_tracks (df :pd .DataFrame )->List [str ]:
    tracks =sorted (df ["trackId"].dropna ().astype (str ).unique ().tolist ())
    return tracks 


    # -----------------------------
    # Build candidate X batches
    # -----------------------------
def _make_base_row (meta :Dict ,track :str )->Dict :
    """
    Começa do baseline dos medians do treino + onehot do track.
    """
    x_cols =meta ["feature_cols"]
    num_medians =meta .get ("num_medians",{})
    cat_onehot =meta .get ("cat_onehot",{})
    # monta dicionário com tudo
    row ={}
    for c in x_cols :
        if c in num_medians :
            row [c ]=num_medians [c ]
        else :
        # onehot ou algo fora: default 0
            row [c ]=0.0 

            # seta onehot do trackId
    for oh in cat_onehot .get ("trackId",[]):
        row [oh ]=1.0 if oh ==f"trackId_{track}"else 0.0 

    return row 


def _calc_sampling_bounds (df_track :pd .DataFrame ,meta :Dict )->Tuple [Dict ,Dict ]:
    """
    Cria limites (min/max) por coluna numérica baseado em dados REAIS daquela pista.
    Fallback: median +/- 3*IQR.
    """
    x_cols =meta ["feature_cols"]
    num_cols =meta .get ("num_cols",[])
    num_medians =meta .get ("num_medians",{})

    lo ={}
    hi ={}

    # só amostrar coisas que realmente variam e são numéricas
    # (setup + alguns contextos controláveis)
    sample_cols =[c for c in x_cols if c in num_cols and c not in ("window_m","stride_m")]
    # calcula por pista com robustez
    for c in sample_cols :
        s =pd .to_numeric (df_track .get (c ,pd .Series (dtype =float )),errors ="coerce")
        s =s .dropna ()
        if len (s )>=50 :
            q1 =s .quantile (0.25 )
            q3 =s .quantile (0.75 )
            iqr =float (q3 -q1 )if pd .notna (q3 -q1 )else 0.0 
            med =float (s .median ())
            lo [c ]=float (max (s .min (),med -3.0 *iqr ))
            hi [c ]=float (min (s .max (),med +3.0 *iqr ))
        elif c in num_medians :
            med =float (num_medians [c ])
            lo [c ]=med -1.0 
            hi [c ]=med +1.0 
        else :
            lo [c ]=-1.0 
            hi [c ]=1.0 

        if lo [c ]==hi [c ]:
            hi [c ]=lo [c ]+1e-6 

    return lo ,hi 


def _sample_batch (base_row :Dict ,lo :Dict ,hi :Dict ,n :int ,rng :np .random .Generator )->Tuple [pd .DataFrame ,List [str ]]:
    """
    Gera batch de candidatos ao redor do base_row com amostragem uniforme.
    Retorna df com colunas na ordem feature_cols.
    """
    feature_cols =list (base_row .keys ())
    df =pd .DataFrame ([base_row ]*n )

    # amostra só colunas em lo/hi
    for c in lo .keys ():
        df [c ]=rng .uniform (lo [c ],hi [c ],size =n ).astype (np .float32 )

        # garante numérico onde precisa
    for c in feature_cols :
        if c .startswith ("trackId_"):
            df [c ]=pd .to_numeric (df [c ],errors ="coerce").fillna (0 ).astype (np .float32 )
        elif isinstance (df [c ].iloc [0 ],(str ,)):
        # não deveria acontecer, mas se acontecer zera
            df [c ]=0.0 

    return df [feature_cols ],feature_cols 


    # -----------------------------
    # Scoring modes
    # -----------------------------
def _score_mode (yp :dict ,mode :str )->float :
    """
    Score escalar para CEM.
    Quanto MAIOR, melhor.
    """

    us =yp ["understeer_index"]
    os =yp ["oversteer_index"]
    bi =yp ["brake_instability"]
    tr =yp ["traction_loss_exit"]
    sc =yp ["steering_correction_rate"]
    st =yp ["stress_proxy"]
    v =yp ["speed__mean"]

    if mode =="aggressive":
        return (
        +2.5 *v 
        -120.0 *tr 
        -60.0 *us 
        -40.0 *os 
        -35.0 *bi 
        -20.0 *sc 
        -10.0 *st 
        )

    elif mode =="balanced":
        return (
        +2.0 *v 
        -90.0 *tr 
        -50.0 *us 
        -50.0 *os 
        -45.0 *bi 
        -30.0 *sc 
        -20.0 *st 
        )

    elif mode =="safe":
        return (
        +1.4 *v 
        -70.0 *tr 
        -40.0 *us 
        -40.0 *os 
        -60.0 *bi 
        -45.0 *sc 
        -35.0 *st 
        )

    else :
        raise ValueError (f"Modo desconhecido: {mode}")


        # -----------------------------
        # Predict helper
        # -----------------------------
def _predict (model :keras .Model ,X_df :pd .DataFrame ,x_scaler :Dict ,y_scaler :Dict ,y_cols :List [str ])->np .ndarray :
    X =X_df .to_numpy (dtype =np .float32 )
    Xs =_x_transform (X ,x_scaler )
    Ys =model .predict (Xs ,batch_size =4096 ,verbose =0 )
    Y =_y_inverse_transform (Ys ,y_scaler )

    # aplica clip (o treino usou clip; aqui podemos opcionalmente respeitar)
    clip_lo =np .asarray (y_scaler .get ("clip_lo",[]),dtype =np .float32 )
    clip_hi =np .asarray (y_scaler .get ("clip_hi",[]),dtype =np .float32 )
    if clip_lo .shape [0 ]==Y .shape [1 ]and clip_hi .shape [0 ]==Y .shape [1 ]:
        Y =np .clip (Y ,clip_lo ,clip_hi )

    return Y .astype (np .float32 )


    # -----------------------------
    # CEM optimization
    # -----------------------------
def _cem_optimize (
model :keras .Model ,
df_track :pd .DataFrame ,
meta :Dict ,
scalers :Dict ,
track :str ,
mode :str ,
n_iter :int ,
seed :int ,
batch_size :int ,
elite_frac :float ,
n_rounds :int ,
)->Tuple [Dict ,Dict ]:
    rng =np .random .default_rng (seed )

    x_cols =meta ["feature_cols"]
    y_cols =meta ["y_cols"]

    base_row =_make_base_row (meta ,track )
    lo ,hi =_calc_sampling_bounds (df_track ,meta )

    # Inicializa distribuição normal (mu/sigma) baseada nos bounds
    mu =np .array ([base_row [c ]for c in x_cols ],dtype =np .float32 )
    sigma =np .ones_like (mu ,dtype =np .float32 )*0.5 

    # coloca sigma maior onde realmente vamos amostrar
    for i ,c in enumerate (x_cols ):
        if c in lo and c in hi :
            sigma [i ]=max (0.25 *(hi [c ]-lo [c ]),0.05 )
        else :
            sigma [i ]=0.0 

    best =None 
    best_score =-1e18 
    diag ={"rounds":[]}

    x_scaler =scalers ["x_scaler"]
    y_scaler =scalers ["y_scaler"]

    # helper: clamp por bounds
    def clamp_vec (x :np .ndarray )->np .ndarray :
        x2 =x .copy ()
        for i ,c in enumerate (x_cols ):
            if c in lo and c in hi :
                x2 [i ]=float (np .clip (x2 [i ],lo [c ],hi [c ]))
        return x2 

        # rodadas CEM
    n_total =0 
    for r in range (n_rounds ):
        n_batch =min (batch_size ,max (256 ,n_iter //max (1 ,n_rounds )))
        # gera candidatos N(mu, sigma)
        Z =rng .standard_normal ((n_batch ,len (x_cols ))).astype (np .float32 )
        Xcand =(mu [None ,:]+Z *sigma [None ,:]).astype (np .float32 )

        # clamp nos bounds
        for i ,c in enumerate (x_cols ):
            if c in lo and c in hi :
                Xcand [:,i ]=np .clip (Xcand [:,i ],lo [c ],hi [c ])

                # dataframe
        X_df =pd .DataFrame (Xcand ,columns =x_cols )

        # pred
        Y =_predict (model ,X_df ,x_scaler ,y_scaler ,y_cols )

        # score
        scores =np .zeros ((n_batch ,),dtype =np .float32 )
        for i in range (n_batch ):
            yp ={y_cols [j ]:float (Y [i ,j ])for j in range (len (y_cols ))}
            scores [i ]=float (_score_mode (yp ,mode ))

            # elite
        k =max (8 ,int (elite_frac *n_batch ))
        elite_idx =np .argsort (scores )[-k :]
        elite_X =Xcand [elite_idx ]
        elite_scores =scores [elite_idx ]

        # update mu/sigma
        mu_new =elite_X .mean (axis =0 )
        sig_new =elite_X .std (axis =0 )

        # trava sigma em cols não-amostradas
        for i ,c in enumerate (x_cols ):
            if c not in lo :
                sig_new [i ]=0.0 
            else :
                sig_new [i ]=float (np .clip (sig_new [i ],1e-6 ,0.60 *(hi [c ]-lo [c ]+1e-6 )))

        mu =mu_new .astype (np .float32 )
        sigma =sig_new .astype (np .float32 )

        # acha melhor do batch
        i_best =int (np .argmax (scores ))
        if float (scores [i_best ])>best_score :
            best_score =float (scores [i_best ])
            best_x =Xcand [i_best ].copy ()
            best_y =Y [i_best ].copy ()

            best ={
            "x":{x_cols [j ]:_sanitize_float (best_x [j ])for j in range (len (x_cols ))},
            "y":{y_cols [j ]:_sanitize_float (best_y [j ])for j in range (len (y_cols ))},
            "score":float (best_score ),
            }

        n_total +=n_batch 
        diag ["rounds"].append ({
        "round":r ,
        "n_batch":n_batch ,
        "elite_k":k ,
        "best_score_so_far":float (best_score ),
        "elite_score_mean":float (np .mean (elite_scores )),
        })

    if best is None :
        raise RuntimeError ("CEM não retornou best")

    diag ["n_total_candidates"]=int (n_total )
    diag ["track"]=track 
    diag ["mode"]=mode 

    return best ,diag 


    # -----------------------------
    # Main
    # -----------------------------
def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--table",default ="setup_behavior_windows")
    ap .add_argument ("--model_dir",default ="models/champion/setup_surrogate_dl")
    ap .add_argument ("--out_json",default ="data/recommended_setups/setup_recommendations.json")
    ap .add_argument ("--out_table",default ="recommended_setups")
    ap .add_argument ("--n_iter",type =int ,default =6000 )
    ap .add_argument ("--seed",type =int ,default =42 )
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    table =args .table ,
    model_dir =args .model_dir ,
    out_json =args .out_json ,
    out_table =args .out_table ,
    n_iter =args .n_iter ,
    seed =args .seed ,
    drop_and_rebuild =args .drop_and_rebuild ,
    )

    print ("=== GENERATE RECOMMENDED SETUPS (CEM, FULL SETUP EXPORT) ===")
    print (f"sqlite: {cfg.sqlite_path}")
    print (f"table:  {cfg.table}")
    print (f"model:  {cfg.model_dir}")
    print (f"out_json: {cfg.out_json}")
    print (f"out_table: {cfg.out_table}")
    print (f"n_iter: {cfg.n_iter} seed: {cfg.seed} drop_and_rebuild: {cfg.drop_and_rebuild}")

    model_dir =Path (cfg .model_dir )
    out_json =Path (cfg .out_json )
    _ensure_dir (out_json .parent )

    meta =_load_meta (model_dir )
    scalers =_load_scalers (model_dir )
    model =_load_model (model_dir )

    # garante que o modelo espera a dimensão certa
    expected_dim =int (model .input_shape [-1 ])
    feature_cols =meta ["feature_cols"]
    if len (feature_cols )!=expected_dim :
        raise ValueError (f"meta.feature_cols={len(feature_cols)} mas modelo espera {expected_dim}")

    con =_connect (cfg .sqlite_path )
    df =_read_df (con ,cfg .table )
    tracks =_get_tracks (df )
    print (f"[ok] tracks: {tracks}")

    # tabela sqlite
    if cfg .drop_and_rebuild :
        _drop_table (con ,cfg .out_table )
    _create_table (con ,cfg .out_table )

    all_out =[]
    t0 =time .time ()

    # salva x_cols usados pra debug
    x_cols_path =out_json .parent /"x_cols_used_from_training_feature_cols.txt"
    x_cols_path .write_text ("\n".join (feature_cols ),encoding ="utf-8")

    for track in tracks :
        df_track =df .loc [df ["trackId"].astype (str )==str (track )].reset_index (drop =True )
        if len (df_track )<200 :
            print (f"[warn] track={track} tem poucas linhas ({len(df_track)}). Mesmo assim vou gerar.")

        for mode in cfg .modes :
            best ,diag =_cem_optimize (
            model =model ,
            df_track =df_track ,
            meta =meta ,
            scalers =scalers ,
            track =str (track ),
            mode =mode ,
            n_iter =cfg .n_iter ,
            seed =cfg .seed ,
            batch_size =cfg .batch_size ,
            elite_frac =cfg .elite_frac ,
            n_rounds =cfg .n_rounds ,
            )

            x_full ={k :_sanitize_float (v )for k ,v in best ["x"].items ()}
            y_pred ={k :_sanitize_float (v )for k ,v in best ["y"].items ()}

            setup_full ,x_context =_split_x_into_setup_and_context (x_full )

            rec ={
            "trackId":str (track ),
            "mode":mode ,
            "score":float (best ["score"]),
            "n_candidates":int (diag .get ("n_total_candidates",cfg .n_iter )),
            "model_dim":int (len (feature_cols )),
            "feature_cols_file":str (x_cols_path ),

            "y_pred":y_pred ,
            "setup_full":setup_full ,
            "x_full":x_full ,
            "x_context":x_context ,
            "diag":diag ,
            }

            all_out .append (rec )
            _insert_row (con ,cfg .out_table ,rec )
            con .commit ()

            print (f"[ok] track={track:10s} mode={mode:10s} score={rec['score']:.3f} speed={y_pred.get('speed__mean', None)}")

            # escreve JSON final
    payload ={
    "created_at":time .strftime ("%Y-%m-%d %H:%M:%S"),
    "model_dir":str (model_dir ),
    "table":cfg .table ,
    "tracks":tracks ,
    "modes":list (cfg .modes ),
    "n_iter":cfg .n_iter ,
    "seed":cfg .seed ,
    "items":all_out ,
    }
    _write_json (out_json ,payload )

    con .close ()
    t1 =time .time ()
    print (f"[done] wrote: {out_json}")
    print (f"[done] sqlite table: {cfg.out_table}")
    print (f"[done] rows: {len(all_out)} time={t1-t0:.2f}s")


if __name__ =="__main__":
    try :
        gpus =tf .config .list_physical_devices ("GPU")
        for g in gpus :
            tf .config .experimental .set_memory_growth (g ,True )
    except Exception :
        pass 

    main ()