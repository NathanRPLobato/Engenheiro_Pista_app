# arquivo: src/train/setup/build_setup_behavior_windows.py
from __future__ import annotations 

import argparse 
import sqlite3 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple 

import numpy as np 
import pandas as pd 

EPS =1e-9 


# -------------------------
# Config
# -------------------------
@dataclass 
class Cfg :
    sqlite_path :str ="data/refined/telemetry.sqlite"
    telemetry_table :str ="telemetry_flat"
    out_table :str ="setup_behavior_windows"

    # windowing (por distância da volta)
    window_m :float =200.0 
    stride_m :float =100.0 
    min_samples :int =40 

    # filtros básicos
    lap_number_min :int =0 
    require_lap_time_gt0 :bool =True 

    # drop/rebuild
    drop_and_rebuild :bool =False 


    # -------------------------
    # Helpers SQL
    # -------------------------
def _table_or_view_exists (con :sqlite3 .Connection ,name :str )->Tuple [bool ,str ]:
    row =con .execute (
    "SELECT type FROM sqlite_master WHERE name=? AND (type='table' OR type='view')",
    (name ,),
    ).fetchone ()
    if row is None :
        return False ,""
    return True ,str (row [0 ])


def _drop_table_or_view (con :sqlite3 .Connection ,name :str )->None :
    exists ,typ =_table_or_view_exists (con ,name )
    if not exists :
        return 
    if typ =="view":
        con .execute (f"DROP VIEW IF EXISTS {name};")
    else :
        con .execute (f"DROP TABLE IF EXISTS {name};")


def _get_columns (con :sqlite3 .Connection ,table :str )->List [str ]:
    return pd .read_sql (f"PRAGMA table_info({table});",con )["name"].tolist ()


    # -------------------------
    # Metrics / Features
    # -------------------------
def _speed (df :pd .DataFrame )->np .ndarray :
    vx =pd .to_numeric (df .get ("velocity_X"),errors ="coerce").to_numpy (dtype =np .float32 )
    vy =pd .to_numeric (df .get ("velocity_Y"),errors ="coerce").to_numpy (dtype =np .float32 )
    vz =pd .to_numeric (df .get ("velocity_Z"),errors ="coerce").to_numpy (dtype =np .float32 )
    return np .sqrt (vx *vx +vy *vy +vz *vz )


def _safe_numeric (s :pd .Series )->np .ndarray :
    return pd .to_numeric (s ,errors ="coerce").to_numpy (dtype =np .float32 )


def _window_stats (x :np .ndarray )->Dict [str ,float ]:
    x =x [np .isfinite (x )]
    if x .size ==0 :
        return {"mean":np .nan ,"std":np .nan ,"min":np .nan ,"max":np .nan }
    return {
    "mean":float (np .mean (x )),
    "std":float (np .std (x )),
    "min":float (np .min (x )),
    "max":float (np .max (x )),
    }


def _compute_targets (win :pd .DataFrame )->Dict [str ,float ]:
    steer =_safe_numeric (win ["steering"])if "steering"in win else np .array ([],dtype =np .float32 )
    brake =_safe_numeric (win ["brake"])if "brake"in win else np .array ([],dtype =np .float32 )
    throttle =_safe_numeric (win ["throttle"])if "throttle"in win else np .array ([],dtype =np .float32 )

    yaw_rate =_safe_numeric (win ["angular_vel_Z"])if "angular_vel_Z"in win else np .array ([],dtype =np .float32 )
    gfy =_safe_numeric (win ["gforce_Y"])if "gforce_Y"in win else np .array ([],dtype =np .float32 )

    speed =_speed (win )if {"velocity_X","velocity_Y","velocity_Z"}.issubset (win .columns )else np .array ([],dtype =np .float32 )

    sr0 =_safe_numeric (win ["wheel_slip_ratio_0"])if "wheel_slip_ratio_0"in win else np .array ([],dtype =np .float32 )
    sr1 =_safe_numeric (win ["wheel_slip_ratio_1"])if "wheel_slip_ratio_1"in win else np .array ([],dtype =np .float32 )
    sr2 =_safe_numeric (win ["wheel_slip_ratio_2"])if "wheel_slip_ratio_2"in win else np .array ([],dtype =np .float32 )
    sr3 =_safe_numeric (win ["wheel_slip_ratio_3"])if "wheel_slip_ratio_3"in win else np .array ([],dtype =np .float32 )

    t =_safe_numeric (win ["lap_time"])if "lap_time"in win else np .array ([],dtype =np .float32 )

    steer_abs_mean =float (np .nanmean (np .abs (steer )))if steer .size else np .nan 
    yaw_abs_mean =float (np .nanmean (np .abs (yaw_rate )))if yaw_rate .size else np .nan 
    sp_mean =float (np .nanmean (speed ))if speed .size else np .nan 

    yaw_gain =(
    yaw_abs_mean /(steer_abs_mean *sp_mean +EPS )
    if np .isfinite (steer_abs_mean )and np .isfinite (yaw_abs_mean )and np .isfinite (sp_mean )
    else np .nan 
    )

    slip_f =np .nanmean ((np .abs (sr0 )+np .abs (sr1 ))/2.0 )if sr0 .size and sr1 .size else np .nan 
    slip_r =np .nanmean ((np .abs (sr2 )+np .abs (sr3 ))/2.0 )if sr2 .size and sr3 .size else np .nan 
    slip_balance =float (slip_r -slip_f )if np .isfinite (slip_r )and np .isfinite (slip_f )else np .nan 

    understeer =(steer_abs_mean /(yaw_abs_mean +EPS ))if np .isfinite (steer_abs_mean )and np .isfinite (yaw_abs_mean )else np .nan 
    oversteer =(max (0.0 ,slip_balance )*yaw_gain )if np .isfinite (slip_balance )and np .isfinite (yaw_gain )else np .nan 

    if brake .size and yaw_rate .size :
        mask_b =(brake >0.2 )&np .isfinite (brake )&np .isfinite (yaw_rate )
        brake_instab =float (np .nanstd (brake [mask_b ])+np .nanstd (yaw_rate [mask_b ]))if np .any (mask_b )else np .nan 
    else :
        brake_instab =np .nan 

    if throttle .size and brake .size and sr2 .size and sr3 .size :
        rear_slip =(np .abs (sr2 )+np .abs (sr3 ))/2.0 
        mask_t =(throttle >0.7 )&(brake <0.1 )&np .isfinite (rear_slip )
        traction_loss =float (np .nanmean (rear_slip [mask_t ]))if np .any (mask_t )else np .nan 
    else :
        traction_loss =np .nan 

    if steer .size >=3 and t .size ==steer .size :
        dt =np .diff (t )
        ds =np .diff (steer )
        ok =np .isfinite (dt )&np .isfinite (ds )&(dt >1e-4 )
        corr_rate =float (np .nanmean (np .abs (ds [ok ]/dt [ok ])))if np .any (ok )else np .nan 
    else :
        corr_rate =np .nan 

    slip_abs_mean =np .nanmean ([slip_f ,slip_r ])if np .isfinite (slip_f )and np .isfinite (slip_r )else np .nan 
    stress =np .nan 
    if np .isfinite (slip_abs_mean ):
        stress =float (
        (np .nanmean (np .abs (gfy ))if gfy .size else 0.0 )
        +(steer_abs_mean if np .isfinite (steer_abs_mean )else 0.0 )
        +(float (np .nanmean (brake ))if brake .size else 0.0 )
        +slip_abs_mean 
        )

    return {
    "yaw_gain":yaw_gain ,
    "slip_balance":slip_balance ,
    "understeer_index":understeer ,
    "oversteer_index":oversteer ,
    "brake_instability":brake_instab ,
    "traction_loss_exit":traction_loss ,
    "steering_correction_rate":corr_rate ,
    "stress_proxy":stress ,
    "steer_abs__mean":steer_abs_mean ,
    "yaw_abs__mean":yaw_abs_mean ,
    "speed__mean":sp_mean ,
    }


def _compute_feature_block (win :pd .DataFrame )->Dict [str ,float ]:
    out :Dict [str ,float ]={}

    for c in ["throttle","brake","clutch","steering"]:
        if c in win .columns :
            st =_window_stats (_safe_numeric (win [c ]))
            out [f"{c}__mean"]=st ["mean"]
            out [f"{c}__std"]=st ["std"]
            out [f"{c}__max"]=st ["max"]
            out [f"{c}__min"]=st ["min"]

    for c in ["gforce_X","gforce_Y","gforce_Z","angular_vel_Z","angular_acc_Z"]:
        if c in win .columns :
            st =_window_stats (_safe_numeric (win [c ]))
            out [f"{c}__mean"]=st ["mean"]
            out [f"{c}__std"]=st ["std"]
            out [f"{c}__max"]=st ["max"]

    if {"velocity_X","velocity_Y","velocity_Z"}.issubset (win .columns ):
        sp =_speed (win )
        st =_window_stats (sp )
        out ["speed__mean"]=st ["mean"]
        out ["speed__std"]=st ["std"]
        out ["speed__max"]=st ["max"]

    slip_cols =[c for c in win .columns if c .startswith ("wheel_slip_ratio_")]
    if len (slip_cols )>=4 :
        sr =np .vstack ([_safe_numeric (win [c ])for c in slip_cols [:4 ]])
        slip_abs =np .nanmean (np .abs (sr ),axis =0 )
        st =_window_stats (slip_abs )
        out ["slip_abs__mean"]=st ["mean"]
        out ["slip_abs__std"]=st ["std"]
        out ["slip_abs__max"]=st ["max"]

    for base in ["tyre_temp_","tyre_press_"]:
        cols =[c for c in win .columns if c .startswith (base )]
        if cols :
            arr =np .vstack ([_safe_numeric (win [c ])for c in cols ])
            out [f"{base}mean"]=float (np .nanmean (arr ))

    return out 


    # -------------------------
    # Windowing by distance
    # -------------------------
def _iter_windows_by_distance (lap_df :pd .DataFrame ,window_m :float ,stride_m :float )->List [Tuple [float ,float ,pd .DataFrame ]]:
    d =pd .to_numeric (lap_df ["lap_distance"],errors ="coerce").to_numpy (dtype =np .float32 )
    ok =np .isfinite (d )
    if not np .any (ok ):
        return []

    dmin =float (np .nanmin (d ))
    dmax =float (np .nanmax (d ))
    if not (np .isfinite (dmin )and np .isfinite (dmax )and dmax >dmin +1.0 ):
        return []

    wins =[]
    start =dmin 
    while start +window_m <=dmax +1e-6 :
        end =start +window_m 
        mask =(d >=start )&(d <end )
        if np .any (mask ):
            wins .append ((start ,end ,lap_df .loc [mask ]))
        start +=stride_m 
    return wins 


def build_setup_behavior_windows (db_path :Path ,cfg :Cfg )->None :
    t0 =time .time ()
    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")

    cols =_get_columns (con ,cfg .telemetry_table )
    required =["race_id","trackId","lap_number","lap_time","lap_distance"]
    missing =[c for c in required if c not in cols ]
    if missing :
        raise SystemExit (f"Faltam colunas obrigatórias em {cfg.telemetry_table}: {missing}")

    setup_cols =[
    "wing_setup_0","wing_setup_1",
    "diff_onThrottle_setup","diff_offThrottle_setup",
    "camber_setup_0","camber_setup_1","camber_setup_2","camber_setup_3",
    "toe_setup_0","toe_setup_1","toe_setup_2","toe_setup_3",
    "susp_spring_setup_0","susp_spring_setup_1","susp_spring_setup_2","susp_spring_setup_3",
    "arb_setup_0","arb_setup_1",
    "susp_height_setup_0","susp_height_setup_1","susp_height_setup_2","susp_height_setup_3",
    "brake_press_setup","brake_bias_setup","brake_engine_setup",
    "tyre_press_setup_0","tyre_press_setup_1","tyre_press_setup_2","tyre_press_setup_3",
    "ballast_setup","fuel_setup",
    "traction_ctrl_setup","abs_setup",
    "front_brake_bias",
    ]
    setup_cols =[c for c in setup_cols if c in cols ]

    metric_cols =[
    "throttle","brake","clutch","steering",
    "velocity_X","velocity_Y","velocity_Z",
    "gforce_X","gforce_Y","gforce_Z",
    "angular_vel_Z","angular_acc_Z",
    "wheel_slip_ratio_0","wheel_slip_ratio_1","wheel_slip_ratio_2","wheel_slip_ratio_3",
    "tyre_temp_0","tyre_temp_1","tyre_temp_2","tyre_temp_3",
    "tyre_press_0","tyre_press_1","tyre_press_2","tyre_press_3",
    ]
    metric_cols =[c for c in metric_cols if c in cols ]

    select_cols =["race_id","trackId","lap_number","lap_time","lap_distance"]+setup_cols +metric_cols 
    seen =set ()
    select_cols =[c for c in select_cols if not (c in seen or seen .add (c ))]

    print ("\n=== BUILD SETUP BEHAVIOR WINDOWS (curve-by-curve) ===")
    print (f"db={db_path}")
    print (f"telemetry={cfg.telemetry_table}")
    print (f"out={cfg.out_table}")
    print (f"window_m={cfg.window_m} stride_m={cfg.stride_m} min_samples={cfg.min_samples}")
    print (f"setup_cols={len(setup_cols)} metric_cols={len(metric_cols)} select_cols={len(select_cols)}")

    if cfg .drop_and_rebuild :
        _drop_table_or_view (con ,cfg .out_table )
        con .commit ()

    races =pd .read_sql (
    f"SELECT DISTINCT race_id FROM {cfg.telemetry_table} ORDER BY race_id;",
    con 
    )["race_id"].astype (str ).tolist ()

    if not races :
        raise SystemExit ("Não achei race_id no telemetry.")

    first_write =True 
    total_rows =0 

    for rid in races :
        where =[f"race_id = ? ",f"lap_number >= {cfg.lap_number_min}"]
        if cfg .require_lap_time_gt0 :
            where .append ("lap_time IS NOT NULL AND lap_time > 0")

        sql =f"""
            SELECT {", ".join(select_cols)}
            FROM {cfg.telemetry_table}
            WHERE {" AND ".join(where)}
            ORDER BY lap_number ASC, lap_time ASC;
        """
        df =pd .read_sql (sql ,con ,params =(rid ,))
        if df .empty :
            continue 

            # normaliza trackId como string
        df ["trackId"]=df ["trackId"].astype (str )

        # setup fixo por corrida (primeiro valor válido)
        setup_vals :Dict [str ,float ]={}
        for c in setup_cols :
            v =pd .to_numeric (df [c ],errors ="coerce")
            setup_vals [c ]=float (v .dropna ().iloc [0 ])if v .notna ().any ()else np .nan 

        for (trackId ,lapn ),lap_df in df .groupby (["trackId","lap_number"],sort =True ):
            lap_df =lap_df .sort_values ("lap_time",kind ="mergesort")

            wins =_iter_windows_by_distance (lap_df ,cfg .window_m ,cfg .stride_m )
            if not wins :
                continue 

            out_rows =[]
            for d0 ,d1 ,wdf in wins :
                if len (wdf )<cfg .min_samples :
                    continue 

                row :Dict [str ,object ]={
                "race_id":str (rid ),
                "trackId":str (trackId ),
                "lap_number":int (lapn )if pd .notna (lapn )else None ,
                "d0":float (d0 ),
                "d1":float (d1 ),
                "window_m":float (cfg .window_m ),
                "stride_m":float (cfg .stride_m ),
                "n_samples":int (len (wdf )),
                "t_start":float (pd .to_numeric (wdf ["lap_time"],errors ="coerce").min ()),
                "t_end":float (pd .to_numeric (wdf ["lap_time"],errors ="coerce").max ()),
                }

                row .update (setup_vals )
                row .update (_compute_feature_block (wdf ))
                row .update (_compute_targets (wdf ))

                out_rows .append (row )

            if not out_rows :
                continue 

            out_df =pd .DataFrame (out_rows )
            out_df .to_sql (cfg .out_table ,con ,if_exists ="replace"if first_write else "append",index =False )
            first_write =False 
            total_rows +=len (out_df )

        con .commit ()
        print (f"[ok] race_id={rid} -> windows acumuladas: {total_rows}")

    if total_rows >0 :
        try :
            con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}__race ON {cfg.out_table}(race_id);")
            con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}__track ON {cfg.out_table}(trackId);")
            con .execute (f"CREATE INDEX IF NOT EXISTS idx_{cfg.out_table}__lap ON {cfg.out_table}(race_id, lap_number);")
            con .commit ()
        except Exception as e :
            print (f"[warn] falha criando índices: {e}")

    con .close ()
    t1 =time .time ()
    print (f"\n[ok] {cfg.out_table} pronto. rows={total_rows}  time={t1 - t0:.2f}s")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default ="data/refined/telemetry.sqlite")
    ap .add_argument ("--telemetry_table",default ="telemetry_flat")
    ap .add_argument ("--out_table",default ="setup_behavior_windows")
    ap .add_argument ("--window_m",type =float ,default =200.0 )
    ap .add_argument ("--stride_m",type =float ,default =100.0 )
    ap .add_argument ("--min_samples",type =int ,default =40 )
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    cfg =Cfg (
    sqlite_path =args .sqlite_path ,
    telemetry_table =args .telemetry_table ,
    out_table =args .out_table ,
    window_m =args .window_m ,
    stride_m =args .stride_m ,
    min_samples =args .min_samples ,
    drop_and_rebuild =args .drop_and_rebuild ,
    )

    build_setup_behavior_windows (db_path =Path (cfg .sqlite_path ),cfg =cfg )


if __name__ =="__main__":
    main ()