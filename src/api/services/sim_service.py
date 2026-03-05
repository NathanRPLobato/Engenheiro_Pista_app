# arquivo: src/api/services/sim_service.py
from __future__ import annotations 

import json 
import time 
import uuid 
from pathlib import Path 
from typing import Any ,Dict ,Optional 

from ..schemas import SimulateRequest 


def _safe_mkdir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _now_run_id ()->str :
# curto, único e bom pra URL
    return time .strftime ("%Y%m%d_%H%M%S")+"_"+uuid .uuid4 ().hex [:10 ]


def _save_run_json (run_id :str ,payload :Dict [str ,Any ])->str :
    out_dir =Path ("outputs")/"api_runs"
    _safe_mkdir (out_dir )
    out_path =out_dir /f"{run_id}.json"
    out_path .write_text (json .dumps (payload ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")
    return str (out_path )


def simulate (req :SimulateRequest )->Dict [str ,Any ]:
    """
    Consome o simulador existente (src/sim/simulate_race_fast.py).
    Import aqui dentro pra não “puxar TF” no import da API inteira.
    """
    from src .sim .simulate_race_fast import SimCfg ,simulate as sim_run # type: ignore

    cfg =SimCfg (
    sqlite_path =req .sqlite_path ,
    trackId =req .trackId ,
    race_laps =req .race_laps ,
    setup_mode =req .setup_mode ,
    style_code =req .style_code ,
    allowed_compounds =req .allowed_compounds ,
    two_compounds_rule =req .two_compounds_rule ,
    pit_loss_s =req .pit_loss_s ,
    max_stints =req .max_stints ,
    max_stint_laps =req .max_stint_laps ,
    min_stint_laps =req .min_stint_laps ,
    )

    # teu simulate_race_fast atual tem drift fixo se tiver campo, aplica.
    # se não tiver, ignora sem quebrar.
    if hasattr (cfg ,"drift_s_per_lap"):
        setattr (cfg ,"drift_s_per_lap",float (req .drift_s_per_lap ))

    out =sim_run (cfg )
    return out 


def run_and_optionally_persist (req :SimulateRequest )->Dict [str ,Any ]:
    run_id =_now_run_id ()
    result =simulate (req )

    saved_json :Optional [str ]=None 
    if req .save_json :
        saved_json =_save_run_json (run_id ,{
        "run_id":run_id ,
        "request":req .model_dump (),
        "result":result ,
        })

    return {
    "ok":True ,
    "run_id":run_id ,
    "saved_json":saved_json ,
    "result":result ,
    }


def read_run_json (run_id :str )->Dict [str ,Any ]:
    p =Path ("outputs")/"api_runs"/f"{run_id}.json"
    if not p .exists ():
        raise FileNotFoundError (f"Run não encontrado: {run_id}")
    return json .loads (p .read_text (encoding ="utf-8"))