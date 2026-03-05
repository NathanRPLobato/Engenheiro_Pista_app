# arquivo: src/train/core/build_windows.py
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


SQLITE_PATH =Path ("data/refined/telemetry.sqlite")


@dataclass 
class WindowsConfig :
    sqlite_path :str =str (SQLITE_PATH )
    telemetry_table :str ="telemetry_flat"
    laps_table :str ="laps"
    baselines_table :str ="baselines_track_compound"
    out_table :str ="telemetry_windows"

    window_sec :float =4.0 
    stride_sec :float =2.0 
    min_samples_per_window :int =30 

    drop_and_rebuild :bool =False 


def _connect (db_path :Path )->sqlite3 .Connection :
    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    return con 


def _table_exists (con :sqlite3 .Connection ,name :str )->bool :
    cur =con .execute (
    "SELECT name FROM sqlite_master WHERE (type='table' OR type='view') AND name=?",
    (name ,),
    )
    return cur .fetchone ()is not None 


def _get_cols (con :sqlite3 .Connection ,table :str )->List [str ]:
    df =pd .read_sql (f"PRAGMA table_info({table});",con )
    return df ["name"].tolist ()


def _drop_view_or_table (con :sqlite3 .Connection ,name :str )->None :
    cur =con .execute (
    "SELECT type FROM sqlite_master WHERE name=?;",
    (name ,),
    )
    row =cur .fetchone ()
    if row is None :
        return 

    obj_type =row [0 ]
    if obj_type =="view":
        con .execute (f"DROP VIEW IF EXISTS {name};")
    elif obj_type =="table":
        con .execute (f"DROP TABLE IF EXISTS {name};")
    con .commit ()


def _safe_float (s :pd .Series )->pd .Series :
# converte string p/ float quando possível, sem travar
    return pd .to_numeric (s ,errors ="coerce")


def _speed_from_velocity (df :pd .DataFrame )->np .ndarray :
    vx =_safe_float (df ["velocity_X"]).to_numpy ()
    vy =_safe_float (df ["velocity_Y"]).to_numpy ()
    vz =_safe_float (df ["velocity_Z"]).to_numpy ()
    spd =np .sqrt (vx *vx +vy *vy +vz *vz )
    return spd 


def _slip_abs_mean (df :pd .DataFrame ,slip_cols :List [str ])->np .ndarray :
    if not slip_cols :
        return np .full (len (df ),np .nan ,dtype =np .float32 )

    m =None 
    for c in slip_cols :
        v =np .abs (_safe_float (df [c ]).to_numpy (dtype =np .float32 ,copy =False ))
        if m is None :
            m =v 
        else :
            m =m +v 
    return (m /float (len (slip_cols ))).astype (np .float32 )


def _make_create_table_sql (table :str ,col_types :Dict [str ,str ])->str :
    cols_sql =[]
    for c ,t in col_types .items ():
        cols_sql .append (f"{c} {t}")
    return f"CREATE TABLE IF NOT EXISTS {table} (\n  "+",\n  ".join (cols_sql )+"\n);"


def _insert_many (con :sqlite3 .Connection ,table :str ,df :pd .DataFrame )->None :
    cols =df .columns .tolist ()
    placeholders =",".join (["?"]*len (cols ))
    sql =f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders});"
    con .executemany (sql ,df .itertuples (index =False ,name =None ))
    con .commit ()


def build_windows (db_path :Path ,cfg :WindowsConfig )->None :
    t0_all =time .time ()
    con =_connect (db_path )

    if not _table_exists (con ,cfg .telemetry_table ):
        raise SystemExit (f"Tabela {cfg.telemetry_table} não existe no SQLite.")
    if not _table_exists (con ,cfg .laps_table ):
        raise SystemExit (f"Tabela {cfg.laps_table} não existe no SQLite.")
    if not _table_exists (con ,cfg .baselines_table ):
        raise SystemExit (f"Tabela {cfg.baselines_table} não existe no SQLite.")

    if cfg .drop_and_rebuild :
        _drop_view_or_table (con ,cfg .out_table )

        # proteção contra lixo antigo
    _drop_view_or_table (con ,"_tmp_tel_for_windows")

    tel_cols =_get_cols (con ,cfg .telemetry_table )

    # colunas mínimas que precisamos p/ recortar janelas
    required =["row_id","race_id","trackId","lap_number","lap_time"]
    for r in required :
        if r not in tel_cols :
            raise SystemExit (f"Coluna obrigatória faltando em {cfg.telemetry_table}: {r}")

            # fuel é importante p/ burn, mas se não tiver a gente segue sem burn
    has_fuel ="fuel"in tel_cols 

    # pilot inputs
    pilot_candidates =["throttle","brake","clutch","steering"]
    pilot_cols =[c for c in pilot_candidates if c in tel_cols ]

    # consequencia na pista
    gforce_cols =[c for c in ["gforce_X","gforce_Y","gforce_Z"]if c in tel_cols ]
    vel_cols_ok =all (c in tel_cols for c in ["velocity_X","velocity_Y","velocity_Z"])

    # consequencia no carro
    car_candidates_mean =[c for c in ["gear","rpm","rpm_perc"]if c in tel_cols ]
    wheel_slip_cols =[c for c in ["wheel_slip_ratio_0","wheel_slip_ratio_1","wheel_slip_ratio_2","wheel_slip_ratio_3"]if c in tel_cols ]

    # setup
    setup_candidates =[
    "diff_onThrottle_setup",
    "diff_offThrottle_setup",
    "brake_press_setup",
    "brake_bias_setup",
    "brake_engine_setup",
    "ballast_setup",
    "fuel_setup",
    "traction_ctrl_setup",
    "abs_setup",
    ]
    setup_cols =[c for c in setup_candidates if c in tel_cols ]

    print ("\n=== BUILD WINDOWS (fingerprint dataset) ===")
    print (f"db={db_path}")
    print (f"telemetry={cfg.telemetry_table}")
    print (f"laps={cfg.laps_table}")
    print (f"baselines={cfg.baselines_table}")
    print (f"out={cfg.out_table}")
    print (f"window_sec={cfg.window_sec} stride_sec={cfg.stride_sec} min_samples={cfg.min_samples_per_window}")
    print (f"pilot_cols={pilot_cols}")
    print (f"car_mean_cols={car_candidates_mean}")
    print (f"gforce_cols={gforce_cols} has_speed={vel_cols_ok}")
    print (f"setup_cols={len(setup_cols)}  has_fuel={has_fuel}  slip_cols={len(wheel_slip_cols)}")

    # carrega laps (p/ ter compound_id por volta)
    df_laps =pd .read_sql (
    f"""
        SELECT race_id, trackId, lap_number, compound_id, lap_time
        FROM {cfg.laps_table}
        WHERE lap_time IS NOT NULL AND lap_time > 0
        """,
    con ,
    )
    if df_laps .empty :
        raise SystemExit ("Tabela laps está vazia ou sem lap_time válido. Rode build_laps antes.")

        # carrega baselines
    df_base =pd .read_sql (
    f"""
        SELECT trackId, compound_id, baseline_mean, baseline_std
        FROM {cfg.baselines_table}
        """,
    con ,
    )
    if df_base .empty :
        raise SystemExit ("Tabela baselines_track_compound está vazia. Rode baselines antes.")

    base_map :Dict [Tuple [str ,int ],Tuple [float ,float ]]={}
    for row in df_base .itertuples (index =False ):
        base_map [(str (row .trackId ),int (row .compound_id ))]=(float (row .baseline_mean ),float (row .baseline_std ))

        # define schema da tabela de saída
        # ids
    col_types :Dict [str ,str ]={
    "race_id":"TEXT",
    "trackId":"TEXT",
    "lap_number":"INTEGER",
    "compound_id":"INTEGER",
    "t0":"REAL",
    "t1":"REAL",
    "n_samples":"INTEGER",
    "lap_time_ref":"REAL",
    "dt_window":"REAL",
    "baseline_mean":"REAL",
    "baseline_std":"REAL",
    "y_pace":"REAL",
    "y_stress":"REAL",
    }

    if has_fuel :
        col_types .update (
        {
        "fuel_start":"REAL",
        "fuel_end":"REAL",
        "y_burn":"REAL",
        "fuel__min":"REAL",
        "fuel__max":"REAL",
        "fuel__mean":"REAL",
        }
        )

        # pilot aggregations
    for c in pilot_cols :
        col_types [f"{c}__mean"]="REAL"
        col_types [f"{c}__max"]="REAL"
        col_types [f"{c}__min"]="REAL"

        # car mean cols
    for c in car_candidates_mean :
        col_types [f"{c}__mean"]="REAL"

        # gforce abs mean
    for c in gforce_cols :
        col_types [f"{c}_abs__mean"]="REAL"

        # speed features
    if vel_cols_ok :
        col_types ["speed__mean"]="REAL"
        col_types ["speed__max"]="REAL"

        # slip features
    if wheel_slip_cols :
        col_types ["slip_abs__mean"]="REAL"
        col_types ["slip_abs__max"]="REAL"

        # setup means
    for c in setup_cols :
        col_types [c ]="REAL"

        # cria tabela saída
    con .execute (_make_create_table_sql (cfg .out_table ,col_types ))
    con .commit ()

    # loop por volta, mas puxando telemetry por race p/ ficar rápido
    races =sorted (df_laps ["race_id"].astype (str ).unique ().tolist ())

    total_windows =0 
    t_read =0.0 
    t_feat =0.0 

    # colunas que vamos selecionar do telemetry_flat
    select_cols =["race_id","trackId","lap_number","lap_time"]
    if has_fuel :
        select_cols .append ("fuel")

    for c in pilot_cols :
        select_cols .append (c )
    for c in car_candidates_mean :
        select_cols .append (c )
    for c in gforce_cols :
        select_cols .append (c )
    if vel_cols_ok :
        select_cols +=["velocity_X","velocity_Y","velocity_Z"]
    for c in wheel_slip_cols :
        select_cols .append (c )
    for c in setup_cols :
        select_cols .append (c )

    select_cols_sql =", ".join (select_cols )

    # limpeza p/ evitar acumular no banco antigo
    if cfg .drop_and_rebuild :
        con .execute (f"DELETE FROM {cfg.out_table};")
        con .commit ()

    for rid in races :
        laps_r =df_laps [df_laps ["race_id"].astype (str )==rid ].copy ()
        if laps_r .empty :
            continue 

            # lê telemetria da corrida, só colunas necessárias
        t1 =time .time ()
        df_tel =pd .read_sql (
        f"""
            SELECT {select_cols_sql}
            FROM {cfg.telemetry_table}
            WHERE race_id = ?
              AND lap_number >= 0
              AND lap_time IS NOT NULL AND lap_time > 0
            ORDER BY lap_number ASC, lap_time ASC
            """,
        con ,
        params =(rid ,),
        )
        t2 =time .time ()
        t_read +=(t2 -t1 )

        if df_tel .empty :
            continue 

            # força tipos básicos
        df_tel ["lap_number"]=pd .to_numeric (df_tel ["lap_number"],errors ="coerce").astype ("Int64")
        df_tel ["lap_time"]=pd .to_numeric (df_tel ["lap_time"],errors ="coerce")

        if has_fuel :
            df_tel ["fuel"]=pd .to_numeric (df_tel ["fuel"],errors ="coerce")

            # por lap
        out_rows =[]
        t3 =time .time ()

        for lap_row in laps_r .itertuples (index =False ):
            lap_number =int (lap_row .lap_number )
            trackId =str (lap_row .trackId )
            compound_id =int (lap_row .compound_id )

            key =(trackId ,compound_id )
            if key not in base_map :
            # sem baseline, pula
                continue 

            baseline_mean ,baseline_std =base_map [key ]

            # recorta telemetria da volta
            g =df_tel [df_tel ["lap_number"].astype ("Int64")==lap_number ]
            if g .empty :
                continue 

            lap_time_ref =float (np .nanmax (g ["lap_time"].to_numpy ()))
            if not np .isfinite (lap_time_ref )or lap_time_ref <=0 :
                continue 

                # gera janelas
                # t0 vai de 0 até (lap_time_ref - window_sec)
            last_t0 =lap_time_ref -cfg .window_sec 
            if last_t0 <=0 :
                continue 

            t0_list =np .arange (0.0 ,last_t0 +1e-9 ,cfg .stride_sec ,dtype =np .float32 )

            lap_time_arr =g ["lap_time"].to_numpy (dtype =np .float32 ,copy =False )

            # pré computa speed e slip por sample, se existirem
            if vel_cols_ok :
                speed_arr =_speed_from_velocity (g ).astype (np .float32 )
            else :
                speed_arr =None 

            if wheel_slip_cols :
                slip_arr =_slip_abs_mean (g ,wheel_slip_cols )
            else :
                slip_arr =None 

                # arrays auxiliares p/ stress
            if "gforce_Y"in gforce_cols :
                gfy =np .abs (_safe_float (g ["gforce_Y"]).to_numpy (dtype =np .float32 ,copy =False ))
            else :
                gfy =None 

            if "steering"in pilot_cols :
                steer_abs =np .abs (_safe_float (g ["steering"]).to_numpy (dtype =np .float32 ,copy =False ))
            else :
                steer_abs =None 

            if "brake"in pilot_cols :
                brake_arr =_safe_float (g ["brake"]).to_numpy (dtype =np .float32 ,copy =False )
            else :
                brake_arr =None 

                # loop de janelas
            for t0w in t0_list :
                t1w =float (t0w +cfg .window_sec )

                mask =(lap_time_arr >=t0w )&(lap_time_arr <t1w )
                idx =np .where (mask )[0 ]
                n_samples =int (idx .size )
                if n_samples <cfg .min_samples_per_window :
                    continue 

                lap_time_start =float (lap_time_arr [idx [0 ]])
                lap_time_end =float (lap_time_arr [idx [-1 ]])
                dt_window =float (lap_time_end -lap_time_start )
                if not np .isfinite (dt_window )or dt_window <=0 :
                    continue 

                row ={
                "race_id":rid ,
                "trackId":trackId ,
                "lap_number":lap_number ,
                "compound_id":compound_id ,
                "t0":float (t0w ),
                "t1":float (t1w ),
                "n_samples":n_samples ,
                "lap_time_ref":lap_time_ref ,
                "dt_window":dt_window ,
                "baseline_mean":float (baseline_mean ),
                "baseline_std":float (baseline_std ),
                }

                # y_pace: diferença do dt_window p/ o esperado pela baseline
                # esperado por janela = baseline_mean (window_sec / lap_time_ref)
                expected =float (baseline_mean )*float (cfg .window_sec /lap_time_ref )
                row ["y_pace"]=float (dt_window -expected )

                # fuel e burn
                if has_fuel :
                    fuel_start =g ["fuel"].to_numpy (dtype =np .float32 ,copy =False )[idx [0 ]]
                    fuel_end =g ["fuel"].to_numpy (dtype =np .float32 ,copy =False )[idx [-1 ]]
                    row ["fuel_start"]=float (fuel_start )if np .isfinite (fuel_start )else None 
                    row ["fuel_end"]=float (fuel_end )if np .isfinite (fuel_end )else None 
                    if np .isfinite (fuel_start )and np .isfinite (fuel_end ):
                        row ["y_burn"]=float (fuel_start -fuel_end )
                    else :
                        row ["y_burn"]=None 

                    fwin =g ["fuel"].to_numpy (dtype =np .float32 ,copy =False )[idx ]
                    row ["fuel__min"]=float (np .nanmin (fwin ))
                    row ["fuel__max"]=float (np .nanmax (fwin ))
                    row ["fuel__mean"]=float (np .nanmean (fwin ))

                    # stress proxy
                    # ideia: combinação de carga lateral, comando, frenagem e slip
                parts =[]
                if gfy is not None :
                    parts .append (float (np .nanmean (gfy [idx ])))
                if steer_abs is not None :
                    parts .append (float (np .nanmean (steer_abs [idx ])))
                if brake_arr is not None :
                    parts .append (float (np .nanmean (brake_arr [idx ])))
                if slip_arr is not None :
                    parts .append (float (np .nanmean (slip_arr [idx ])))

                row ["y_stress"]=float (np .sum (parts ))if parts else None 

                # pilot aggregations
                for c in pilot_cols :
                    v =_safe_float (g [c ]).to_numpy (dtype =np .float32 ,copy =False )[idx ]
                    row [f"{c}__mean"]=float (np .nanmean (v ))
                    row [f"{c}__max"]=float (np .nanmax (v ))
                    row [f"{c}__min"]=float (np .nanmin (v ))

                    # car mean cols
                for c in car_candidates_mean :
                    v =_safe_float (g [c ]).to_numpy (dtype =np .float32 ,copy =False )[idx ]
                    row [f"{c}__mean"]=float (np .nanmean (v ))

                    # gforce abs mean
                for c in gforce_cols :
                    v =np .abs (_safe_float (g [c ]).to_numpy (dtype =np .float32 ,copy =False ))[idx ]
                    row [f"{c}_abs__mean"]=float (np .nanmean (v ))

                    # speed
                if speed_arr is not None :
                    sw =speed_arr [idx ]
                    row ["speed__mean"]=float (np .nanmean (sw ))
                    row ["speed__max"]=float (np .nanmax (sw ))

                    # slip
                if slip_arr is not None :
                    sl =slip_arr [idx ]
                    row ["slip_abs__mean"]=float (np .nanmean (sl ))
                    row ["slip_abs__max"]=float (np .nanmax (sl ))

                    # setup means
                for c in setup_cols :
                    v =_safe_float (g [c ]).to_numpy (dtype =np .float32 ,copy =False )[idx ]
                    row [c ]=float (np .nanmean (v ))

                out_rows .append (row )

        t4 =time .time ()
        t_feat +=(t4 -t3 )

        if out_rows :
            df_out =pd .DataFrame (out_rows )

            # garante as colunas da tabela
            for c in col_types .keys ():
                if c not in df_out .columns :
                    df_out [c ]=None 

            df_out =df_out [list (col_types .keys ())]
            _insert_many (con ,cfg .out_table ,df_out )
            total_windows +=len (df_out )

            # salva features.json p/ o autoencoder e p/ modelos tabulares
    feature_cols =[]
    for c in col_types .keys ():
        if c in ["race_id","trackId","lap_number","compound_id","t0","t1"]:
            continue 
        feature_cols .append (c )

    outdir =Path ("data/feature_store/fingerprint")
    outdir .mkdir (parents =True ,exist_ok =True )
    features_json ={
    "table":cfg .out_table ,
    "features":feature_cols ,
    "categorical":["trackId","compound_id"],
    "labels":["y_pace","y_stress","y_burn"]if has_fuel else ["y_pace","y_stress"],
    "window_sec":cfg .window_sec ,
    "stride_sec":cfg .stride_sec ,
    "min_samples_per_window":cfg .min_samples_per_window ,
    }
    (outdir /"features.json").write_text (json .dumps (features_json ,indent =2 ),encoding ="utf-8")

    con .close ()
    t_end =time .time ()
    print ("\n=== SUMMARY WINDOWS ===")
    print (f"windows_total={total_windows}")
    print (f"time_read_sec={t_read:.2f}")
    print (f"time_feat_sec={t_feat:.2f}")
    print (f"time_total_sec={(t_end - t0_all):.2f}")
    print (f"[ok] gravado: {cfg.out_table}")
    print (f"[ok] features.json: {outdir / 'features.json'}")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (SQLITE_PATH ))
    ap .add_argument ("--telemetry_table",default ="telemetry_flat")
    ap .add_argument ("--laps_table",default ="laps")
    ap .add_argument ("--baselines_table",default ="baselines_track_compound")
    ap .add_argument ("--out_table",default ="telemetry_windows")

    ap .add_argument ("--window_sec",type =float ,default =4.0 )
    ap .add_argument ("--stride_sec",type =float ,default =2.0 )
    ap .add_argument ("--min_samples_per_window",type =int ,default =30 )

    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    args =ap .parse_args ()

    cfg =WindowsConfig (
    sqlite_path =args .sqlite_path ,
    telemetry_table =args .telemetry_table ,
    laps_table =args .laps_table ,
    baselines_table =args .baselines_table ,
    out_table =args .out_table ,
    window_sec =float (args .window_sec ),
    stride_sec =float (args .stride_sec ),
    min_samples_per_window =int (args .min_samples_per_window ),
    drop_and_rebuild =bool (args .drop_and_rebuild ),
    )

    build_windows (db_path =Path (cfg .sqlite_path ),cfg =cfg )


if __name__ =="__main__":
    main ()
