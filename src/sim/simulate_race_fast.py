# arquivo: src/sim/simulate_race_fast.py
from __future__ import annotations 

import argparse 
import json 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Tuple 

import numpy as np 
import pandas as pd 

# opcional: TF só pra não quebrar seu ambiente (não usamos inferência aqui)
import tensorflow as tf # noqa: F401


# -----------------------------
# CONFIG
# -----------------------------
@dataclass 
class SimCfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"

    # inputs principais
    trackId :str ="Montreal"
    race_laps :int =50 

    # setup
    setup_mode :str ="balanced"# balanced/aggressive/conservative
    style_code :str ="STYLE_BALANCED"

    # pneus
    allowed_compounds :List [int ]=None 
    two_compounds_rule :bool =True 

    # pit
    pit_loss_s :float =22.0 
    max_stints :int =25 
    max_stint_laps :int =25 
    min_stint_laps :int =5 

    # artifacts
    setup_reco_json :str ="data/recommended_setups/setup_recommendations.json"

    # performance
    use_cache_in_process :bool =True 
    drift_s_per_lap :float =0.002 # drift leve por volta (ajustável)


    # -----------------------------
    # IN-PROCESS CACHE
    # -----------------------------
_CACHE :Dict [str ,Any ]={}


def _cache_get (key :str ):
    return _CACHE .get (key ,None )


def _cache_set (key :str ,val :Any ):
    _CACHE [key ]=val 
    return val 


    # -----------------------------
    # SQLITE
    # -----------------------------
def _connect (db :str )->sqlite3 .Connection :
# timeout maior pra evitar erro bobo em ambiente com WAL
    con =sqlite3 .connect (db ,timeout =30 )
    # PRAGMAs p/ leitura rápida
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    con .execute ("PRAGMA temp_store=MEMORY;")
    con .execute ("PRAGMA cache_size=-200000;")# 200MB se puder (ajusta se RAM baixa)
    return con 


def _read_json (path :Path )->Dict [str ,Any ]:
    return json .loads (path .read_text (encoding ="utf-8"))


def _ensure_list_int (s :str )->List [int ]:
    if s is None or str (s ).strip ()=="":
        return []
    parts =[p .strip ()for p in str (s ).split (",")if p .strip ()!=""]
    return [int (float (p ))for p in parts ]


    # -----------------------------
    # SETUP RECOMMENDATION
    # -----------------------------
def _load_setup_recommendations (cfg :SimCfg )->List [Dict [str ,Any ]]:
    key =f"setup_recos::{cfg.setup_reco_json}"
    if cfg .use_cache_in_process :
        cached =_cache_get (key )
        if cached is not None :
            return cached 

    p =Path (cfg .setup_reco_json )
    if not p .exists ():
        raise SystemExit (f"setup_recommendations.json não existe em: {p}")

    data =json .loads (p .read_text (encoding ="utf-8"))
    if isinstance (data ,dict )and "items"in data :
        items =data ["items"]
    elif isinstance (data ,list ):
        items =data 
    else :
        raise SystemExit ("setup_recommendations.json: formato inesperado")

    if cfg .use_cache_in_process :
        return _cache_set (key ,items )
    return items 


def _pick_setup_for_track (items :List [Dict [str ,Any ]],trackId :str ,mode :str )->Dict [str ,Any ]:
    best =None 
    for it in items :
        if str (it .get ("trackId"))!=str (trackId ):
            continue 
        if str (it .get ("mode"))!=str (mode ):
            continue 
        if best is None or float (it .get ("score",-1e18 ))>float (best .get ("score",-1e18 )):
            best =it 

    if best is None :
        raise SystemExit (f"Não achei setup recomendado para trackId={trackId} mode={mode}")

    return {
    "trackId":best .get ("trackId"),
    "mode":best .get ("mode"),
    "score":best .get ("score"),
    "setup_full":best .get ("setup_full",best .get ("setup",{})),
    "y_pred":best .get ("y_pred",{}),
    }


    # -----------------------------
    # BASELINE LAP TIME
    # -----------------------------
def _load_baselines_map (con :sqlite3 .Connection ,trackId :str )->Dict [int ,float ]:
    """
    Retorna dict compound_id -> baseline_mean (lap_time médio)
    """
    key =f"baselines::{trackId}"
    cached =_cache_get (key )
    if cached is not None :
        return cached 

    df =pd .read_sql (
    """
        SELECT compound_id, baseline_mean
        FROM baselines_track_compound
        WHERE trackId = ?
        """,
    con ,
    params =(str (trackId ),),
    )

    out :Dict [int ,float ]={}
    if not df .empty :
        df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce")
        df ["baseline_mean"]=pd .to_numeric (df ["baseline_mean"],errors ="coerce")
        df =df .dropna (subset =["compound_id","baseline_mean"])
        for r in df .itertuples (index =False ):
            out [int (r .compound_id )]=float (r .baseline_mean )

            # fallback geral da pista (média dos compounds existentes)
    if len (out )==0 :
        df2 =pd .read_sql (
        """
            SELECT AVG(baseline_mean) AS baseline_mean
            FROM baselines_track_compound
            WHERE trackId = ?
            """,
        con ,
        params =(str (trackId ),),
        )
        base =float (df2 .loc [0 ,"baseline_mean"])if (not df2 .empty and pd .notna (df2 .loc [0 ,"baseline_mean"]))else 90.0 
        out ={"__track_fallback__":base }# marcador

    return _cache_set (key ,out )


def _baseline_for_compound (bmap :Dict [int ,float ],compound_id :int )->float :
    if compound_id in bmap :
        return float (bmap [compound_id ])
        # fallback por pista (se existir)
    fb =bmap .get ("__track_fallback__",None )
    if fb is not None :
        return float (fb )
        # fallback médio dos que existem
    if len (bmap )>0 :
        vals =[v for k ,v in bmap .items ()if isinstance (k ,int )]
        if vals :
            return float (np .mean (vals ))
    return 90.0 


    # -----------------------------
    # DEGRADAÇÃO (data-driven a partir de lap_degradation_windows)
    # -----------------------------
def _deg_penalties_from_db (
cfg :SimCfg ,
con :sqlite3 .Connection ,
trackId :str ,
compounds :List [int ],
race_laps :int ,
)->Dict [int ,np .ndarray ]:
    """
    1 query só: puxa (compound_id, lap_number, y_pace_window, y_stress_window)
    Condensa por lap_number (mediana).
    Produz penalty por volta (segundos), com drift leve.
    """
    compounds =[int (c )for c in compounds ]
    key =f"deg::track={trackId}::comps={','.join(map(str,compounds))}::laps={race_laps}::drift={cfg.drift_s_per_lap}"
    cached =_cache_get (key )
    if cached is not None :
        return cached 

    if not compounds :
        return {}

        # placeholders IN (, , )
    qs =",".join (["?"]*len (compounds ))
    df =pd .read_sql (
    f"""
        SELECT compound_id, lap_number, y_pace_window, y_stress_window
        FROM lap_degradation_windows
        WHERE trackId = ?
          AND compound_id IN ({qs})
        """,
    con ,
    params =[str (trackId )]+compounds ,
    )

    out :Dict [int ,np .ndarray ]={}
    if df .empty :
        for c in compounds :
            out [c ]=np .zeros ((race_laps ,),dtype =np .float32 )
        return _cache_set (key ,out )

    df ["compound_id"]=pd .to_numeric (df ["compound_id"],errors ="coerce")
    df ["lap_number"]=pd .to_numeric (df ["lap_number"],errors ="coerce")
    df ["y_pace_window"]=pd .to_numeric (df ["y_pace_window"],errors ="coerce")
    df ["y_stress_window"]=pd .to_numeric (df ["y_stress_window"],errors ="coerce")
    df =df .dropna (subset =["compound_id","lap_number","y_pace_window","y_stress_window"]).copy ()
    df ["compound_id"]=df ["compound_id"].astype (int )
    df ["lap_number"]=df ["lap_number"].astype (int )

    drift =cfg .drift_s_per_lap *np .arange (race_laps ,dtype =np .float32 )

    for comp in compounds :
        dfi =df [df ["compound_id"]==comp ]
        if dfi .empty :
            out [comp ]=np .zeros ((race_laps ,),dtype =np .float32 )
            continue 

        per_lap =(
        dfi .groupby ("lap_number",as_index =False )
        .agg (pace_med =("y_pace_window","median"),stress_med =("y_stress_window","median"))
        .sort_values ("lap_number")
        )

        pace =per_lap ["pace_med"].to_numpy (dtype =np .float32 )
        stress =per_lap ["stress_med"].to_numpy (dtype =np .float32 )

        # penalidade: só positiva
        base_pen =np .maximum (pace ,0.0 )*(1.0 +0.15 *np .clip (stress ,0.0 ,3.0 ))

        laps_avail =per_lap ["lap_number"].to_numpy (dtype =int )
        arr =np .zeros ((race_laps ,),dtype =np .float32 )

        # preenche por lap_number; se faltar lap, usa último disponível (nearest past)
        last_idx =0 
        for i in range (race_laps ):
            lap =i +1 
            # avança enquanto tiver lap <= atual
            while last_idx +1 <len (laps_avail )and laps_avail [last_idx +1 ]<=lap :
                last_idx +=1 
                # se o primeiro disponível é maior que lap atual, usa o primeiro
            if laps_avail [0 ]>lap :
                arr [i ]=base_pen [0 ]
            else :
                arr [i ]=base_pen [last_idx ]

        arr =arr +drift 
        out [comp ]=arr .astype (np .float32 )

    return _cache_set (key ,out )


    # -----------------------------
    # STRATEGY (DP) - rápido e com constraints
    # -----------------------------
def _dp_best_strategy (
race_laps :int ,
allowed_compounds :List [int ],
two_compounds_rule :bool ,
pit_loss_s :float ,
min_stint_laps :int ,
max_stint_laps :int ,
max_stints :int ,
baseline_by_comp :Dict [int ,float ],
penalties_by_comp :Dict [int ,np .ndarray ],
)->Tuple [float ,List [Tuple [int ,int ]]]:
    """
    DP em cima de "stint como bloco".
    Estado: dp[pos][mask] = menor tempo para completar pos voltas usando conjunto mask de compounds.
    Transição: escolhe compound c e length k (min..max), adiciona custo do stint + pit loss (se pos>0).
    Reconstrói sequência de (compound, laps).
    """

    comps =[int (c )for c in allowed_compounds ]
    if not comps :
        raise SystemExit ("allowed_compounds vazio.")

        # map compound -> bit
    bit ={c :1 <<i for i ,c in enumerate (comps )}
    full_masks =range (1 <<len (comps ))

    # prefix sums de penalty por comp
    pen_ps :Dict [int ,np .ndarray ]={}
    for c in comps :
        pen =penalties_by_comp .get (c ,None )
        if pen is None or len (pen )<race_laps :
            pen =np .zeros ((race_laps ,),dtype =np .float32 )
        ps =np .zeros ((race_laps +1 ,),dtype =np .float32 )
        ps [1 :]=np .cumsum (pen [:race_laps ],dtype =np .float32 )
        pen_ps [c ]=ps 

        # dp[pos][mask] = best time
    INF =1e18 
    dp =np .full ((race_laps +1 ,1 <<len (comps )),INF ,dtype =np .float64 )
    prev =[[None for _ in full_masks ]for _ in range (race_laps +1 )]
    stints_used =np .full ((race_laps +1 ,1 <<len (comps )),10 **9 ,dtype =np .int32 )

    dp [0 ,0 ]=0.0 
    stints_used [0 ,0 ]=0 

    for pos in range (race_laps ):
        for mask in full_masks :
            cur =float (dp [pos ,mask ])
            if cur >=INF :
                continue 
            cur_st =int (stints_used [pos ,mask ])
            if cur_st >=max_stints :
                continue 

                # tenta adicionar um stint
            for c in comps :
                b =_baseline_for_compound (baseline_by_comp ,c )
                ps =pen_ps [c ]
                for k in range (min_stint_laps ,max_stint_laps +1 ):
                    nxt =pos +k 
                    if nxt >race_laps :
                        break 
                        # custo do stint = baselinek + sum(pen[pos:pos+k])
                    pen_sum =float (ps [nxt ]-ps [pos ])
                    stint_cost =b *k +pen_sum 
                    add_pit =pit_loss_s if pos >0 else 0.0 
                    new_cost =cur +stint_cost +add_pit 
                    new_mask =mask |bit [c ]
                    new_st =cur_st +1 

                    if new_cost <dp [nxt ,new_mask ]:
                        dp [nxt ,new_mask ]=new_cost 
                        stints_used [nxt ,new_mask ]=new_st 
                        prev [nxt ][new_mask ]=(pos ,mask ,c ,k )

                        # escolhe melhor mask final respeitando regra 2 compounds
    best_total =INF 
    best_mask =None 
    for mask in full_masks :
        if two_compounds_rule and bin (mask ).count ("1")<2 :
            continue 
        val =float (dp [race_laps ,mask ])
        if val <best_total :
            best_total =val 
            best_mask =mask 

    if best_mask is None or best_total >=INF :
        raise SystemExit ("Não achei estratégia válida com constraints atuais.")

        # reconstrói
    plan :List [Tuple [int ,int ]]=[]
    pos =race_laps 
    mask =best_mask 
    while pos >0 :
        p =prev [pos ][mask ]
        if p is None :
        # não deveria acontecer
            break 
        pos0 ,mask0 ,c ,k =p 
        plan .append ((int (c ),int (k )))
        pos ,mask =pos0 ,mask0 

    plan .reverse ()
    return float (best_total ),plan 


def _build_breakdown (
trackId :str ,
plan :List [Tuple [int ,int ]],
baseline_by_comp :Dict [int ,float ],
penalties_by_comp :Dict [int ,np .ndarray ],
pit_loss_s :float ,
)->Tuple [int ,List [Dict [str ,Any ]]]:
    breakdown :List [Dict [str ,Any ]]=[]
    lap_cursor =0 
    for i ,(comp ,n_laps )in enumerate (plan ,start =1 ):
        b =_baseline_for_compound (baseline_by_comp ,comp )
        pen =penalties_by_comp .get (comp ,np .zeros ((lap_cursor +n_laps ,),dtype =np .float32 ))

        seg =pen [lap_cursor :lap_cursor +n_laps ]
        if len (seg )<n_laps :
        # estende
            last =float (seg [-1 ])if len (seg )else 0.0 
            seg =np .concatenate ([seg ,np .full ((n_laps -len (seg ),),last ,dtype =np .float32 )])

        stint_time =float (n_laps *b +float (np .sum (seg )))
        breakdown .append (
        {
        "stint_index":i ,
        "compound_id":int (comp ),
        "laps":int (n_laps ),
        "stint_time_s":stint_time ,
        "pace_summary":{
        "baseline_mean_lap_s":float (b ),
        "avg_deg_penalty_s":float (np .mean (seg ))if len (seg )else 0.0 ,
        "max_deg_penalty_s":float (np .max (seg ))if len (seg )else 0.0 ,
        },
        }
        )
        lap_cursor +=n_laps 

    pit_stops =max (0 ,len (plan )-1 )
    return pit_stops ,breakdown 


    # -----------------------------
    # MAIN SIM
    # -----------------------------
def simulate (cfg :SimCfg )->Dict [str ,Any ]:
    t0 =time .time ()

    if not cfg .allowed_compounds :
        raise SystemExit ("allowed_compounds vazio (ex: 16,17,18).")

        # Setup (JSON)
    setup_items =_load_setup_recommendations (cfg )
    setup_pick =_pick_setup_for_track (setup_items ,cfg .trackId ,cfg .setup_mode )

    # SQLite único
    con =_connect (cfg .sqlite_path )

    # Baselines
    base_map =_load_baselines_map (con ,cfg .trackId )

    # Degradação (1 query só)
    penalties_by_comp =_deg_penalties_from_db (
    cfg =cfg ,
    con =con ,
    trackId =cfg .trackId ,
    compounds =cfg .allowed_compounds ,
    race_laps =cfg .race_laps ,
    )

    # DP strategy
    best_total ,plan =_dp_best_strategy (
    race_laps =cfg .race_laps ,
    allowed_compounds =cfg .allowed_compounds ,
    two_compounds_rule =cfg .two_compounds_rule ,
    pit_loss_s =cfg .pit_loss_s ,
    min_stint_laps =cfg .min_stint_laps ,
    max_stint_laps =cfg .max_stint_laps ,
    max_stints =cfg .max_stints ,
    baseline_by_comp =base_map ,
    penalties_by_comp =penalties_by_comp ,
    )

    con .close ()

    pit_stops ,breakdown =_build_breakdown (
    trackId =cfg .trackId ,
    plan =plan ,
    baseline_by_comp =base_map ,
    penalties_by_comp =penalties_by_comp ,
    pit_loss_s =cfg .pit_loss_s ,
    )

    out ={
    "ok":True ,
    "resolved":{
    "trackId":cfg .trackId ,
    "race_laps":int (cfg .race_laps ),
    "style_code":cfg .style_code ,
    "setup_mode":cfg .setup_mode ,
    "compound_ids_allowed":[int (x )for x in cfg .allowed_compounds ],
    "tyre_rule_two_compounds":bool (cfg .two_compounds_rule ),
    "pit_loss_s":float (cfg .pit_loss_s ),
    "max_stints":int (cfg .max_stints ),
    "max_stint_laps":int (cfg .max_stint_laps ),
    "min_stint_laps":int (cfg .min_stint_laps ),
    "drift_s_per_lap":float (cfg .drift_s_per_lap ),
    },
    "setup":{
    "trackId":setup_pick ["trackId"],
    "mode":setup_pick ["mode"],
    "score":setup_pick ["score"],
    "y_pred":setup_pick ["y_pred"],
    "setup_full":setup_pick ["setup_full"],# só setup de verdade
    },
    "strategy":{
    "stints":[{"compound_id":int (c ),"laps":int (n )}for (c ,n )in plan ],
    "pit_stops":int (pit_stops ),
    "pit_loss_total_s":float (pit_stops *cfg .pit_loss_s ),
    },
    "prediction":{
    "total_time_s":float (best_total ),
    "total_time_min":float (best_total /60.0 ),
    "stints_breakdown":breakdown ,
    },
    "perf":{
    "elapsed_s":float (time .time ()-t0 ),
    "deg_pred_laps":int (cfg .race_laps ),
    "deg_pred_points":int (cfg .race_laps *max (1 ,len (cfg .allowed_compounds ))),
    },
    }
    return out 


    # -----------------------------
    # CLI
    # -----------------------------
def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")

    ap .add_argument ("--trackId",required =True )
    ap .add_argument ("--race_laps",type =int ,required =True )

    ap .add_argument ("--setup_mode",default ="balanced")
    ap .add_argument ("--style_code",default ="STYLE_BALANCED")

    ap .add_argument ("--allowed_compounds",default ="")
    ap .add_argument ("--two_compounds_rule",action ="store_true")

    ap .add_argument ("--pit_loss_s",type =float ,default =22.0 )
    ap .add_argument ("--max_stints",type =int ,default =25 )
    ap .add_argument ("--max_stint_laps",type =int ,default =25 )
    ap .add_argument ("--min_stint_laps",type =int ,default =5 )

    ap .add_argument ("--drift_s_per_lap",type =float ,default =0.002 )

    args =ap .parse_args ()

    cfg =SimCfg (
    sqlite_path =args .sqlite_path ,
    trackId =str (args .trackId ),
    race_laps =int (args .race_laps ),
    setup_mode =str (args .setup_mode ),
    style_code =str (args .style_code ),
    allowed_compounds =_ensure_list_int (args .allowed_compounds ),
    two_compounds_rule =bool (args .two_compounds_rule ),
    pit_loss_s =float (args .pit_loss_s ),
    max_stints =int (args .max_stints ),
    max_stint_laps =int (args .max_stint_laps ),
    min_stint_laps =int (args .min_stint_laps ),
    drift_s_per_lap =float (args .drift_s_per_lap ),
    )

    print (json .dumps (simulate (cfg ),indent =2 ))


if __name__ =="__main__":
    try :
        gpus =tf .config .list_physical_devices ("GPU")
        for g in gpus :
            tf .config .experimental .set_memory_growth (g ,True )
    except Exception :
        pass 

    main ()