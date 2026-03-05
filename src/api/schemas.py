# arquivo: src/api/schemas.py
from __future__ import annotations 

from typing import List ,Optional ,Literal ,Dict ,Any 
from pydantic import BaseModel ,Field 


SetupMode =Literal ["balanced","aggressive","conservative"]


class SimulateRequest (BaseModel ):
    sqlite_path :str =Field (default ="data/refined/telemetry.sqlite")

    trackId :str 
    race_laps :int =Field (ge =1 ,le =400 )

    setup_mode :SetupMode ="balanced"
    style_code :str ="STYLE_BALANCED"

    allowed_compounds :List [int ]=Field (default_factory =list ,description ="Ex: [16,17,18]")
    two_compounds_rule :bool =True 

    pit_loss_s :float =Field (default =22.0 ,ge =0.0 ,le =60.0 )
    max_stints :int =Field (default =25 ,ge =1 ,le =60 )
    max_stint_laps :int =Field (default =25 ,ge =1 ,le =200 )
    min_stint_laps :int =Field (default =5 ,ge =1 ,le =200 )

    # drift simples por volta (p/ dar 'vida' ao pace)
    drift_s_per_lap :float =Field (default =0.002 ,ge =0.0 ,le =0.05 )

    # p/ salvar no disco
    save_json :bool =True 


class SimulateResponse (BaseModel ):
    ok :bool 
    run_id :str 
    saved_json :Optional [str ]=None 
    result :Dict [str ,Any ]


class TelemetrySeriesRequest (BaseModel ):
    sqlite_path :str =Field (default ="data/refined/telemetry.sqlite")
    trackId :str 
    race_id :Optional [str ]=None 
    lap_number :Optional [int ]=None 

    columns :List [str ]=Field (default_factory =lambda :["lap_distance","throttle","brake","steering"])
    limit :int =Field (default =6000 ,ge =100 ,le =200000 )

    # downsample simples pra não explodir payload
    downsample_every :int =Field (default =5 ,ge =1 ,le =200 )