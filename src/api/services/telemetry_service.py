# arquivo: src/api/services/telemetry_service.py
from __future__ import annotations 

from typing import Any ,Dict ,List ,Optional 

import pandas as pd 

from ..db import DBConfig ,connect_db 


def list_tracks (sqlite_path :str )->List [str ]:
    con =connect_db (DBConfig (sqlite_path =sqlite_path ))
    try :
        df =pd .read_sql ("SELECT DISTINCT trackId FROM telemetry_flat ORDER BY trackId;",con )
        return [str (x )for x in df ["trackId"].tolist ()]
    finally :
        con .close ()


def list_races (sqlite_path :str ,trackId :Optional [str ]=None ,limit :int =50 )->List [Dict [str ,Any ]]:
    con =connect_db (DBConfig (sqlite_path =sqlite_path ))
    try :
        if trackId :
            df =pd .read_sql (
            """
                SELECT DISTINCT race_id, trackId
                FROM telemetry_flat
                WHERE trackId = ?
                ORDER BY race_id DESC
                LIMIT ?
                """,
            con ,
            params =(str (trackId ),int (limit )),
            )
        else :
            df =pd .read_sql (
            """
                SELECT DISTINCT race_id, trackId
                FROM telemetry_flat
                ORDER BY race_id DESC
                LIMIT ?
                """,
            con ,
            params =(int (limit ),),
            )
        return df .to_dict (orient ="records")
    finally :
        con .close ()


def telemetry_series (
sqlite_path :str ,
trackId :str ,
columns :List [str ],
race_id :Optional [str ]=None ,
lap_number :Optional [int ]=None ,
limit :int =6000 ,
downsample_every :int =5 ,
)->Dict [str ,Any ]:
    con =connect_db (DBConfig (sqlite_path =sqlite_path ))
    try :
    # valida colunas contra schema real
        cols_df =pd .read_sql ("PRAGMA table_info(telemetry_flat);",con )
        available =set (cols_df ["name"].tolist ())

        safe_cols =[c for c in columns if c in available ]
        if "lap_distance"not in safe_cols and "lap_distance"in available :
            safe_cols =["lap_distance"]+safe_cols 

        if not safe_cols :
            raise ValueError ("Nenhuma coluna válida pedida. Verifique nomes no telemetry_flat.")

        where =["trackId = ?"]
        params :List [Any ]=[str (trackId )]

        if race_id :
            where .append ("race_id = ?")
            params .append (str (race_id ))

        if lap_number is not None :
            where .append ("lap_number = ?")
            params .append (int (lap_number ))

        sql =f"""
        SELECT {", ".join(safe_cols)}
        FROM telemetry_flat
        WHERE {" AND ".join(where)}
        ORDER BY lap_distance
        LIMIT ?
        """
        params .append (int (limit ))

        df =pd .read_sql (sql ,con ,params =tuple (params ))

        if df .empty :
            return {"ok":True ,"rows":0 ,"columns":safe_cols ,"data":[]}

        if downsample_every >1 :
            df =df .iloc [::downsample_every ].copy ()

        return {
        "ok":True ,
        "rows":int (len (df )),
        "columns":safe_cols ,
        "data":df .to_dict (orient ="records"),
        }
    finally :
        con .close ()