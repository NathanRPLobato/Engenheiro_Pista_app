# arquivo: src/api/main.py
from __future__ import annotations 

from fastapi import FastAPI 
from fastapi .middleware .cors import CORSMiddleware 
from pydantic import BaseModel ,Field 
from typing import List ,Optional ,Any ,Dict 

# import seu simulate (ajusta o import se seu path estiver diferente)
from src .sim .simulate_race_fast import SimCfg ,simulate 


app =FastAPI (title ="Race Strategy API",version ="1.0.0")

# -----------------------------
# CORS (resolve OPTIONS 405)
# -----------------------------
# p/ DEV: libera tudo.
# Quando for PROD, troque allow_origins p/ o domínio exato do seu web.
app .add_middleware (
CORSMiddleware ,
allow_origins =["*"],# DEV
allow_credentials =False ,# com '' tem que ser False
allow_methods =["*"],# inclui OPTIONS
allow_headers =["*"],
max_age =86400 ,
)

# -----------------------------
# Request/Response Models
# -----------------------------
class SimulateReq (BaseModel ):
    sqlite_path :str ="data/refined/telemetry.sqlite"
    trackId :str 
    race_laps :int 

    setup_mode :str ="balanced"
    style_code :str ="STYLE_BALANCED"

    allowed_compounds :List [int ]=Field (default_factory =list )
    two_compounds_rule :bool =True 

    pit_loss_s :float =22.0 
    max_stints :int =25 
    max_stint_laps :int =25 
    min_stint_laps :int =5 

    # regra nova
    min_compound_total_laps :int =15 

    # drift
    drift_s_per_lap :float =0.002 


@app .get ("/health")
def health ():
    return {"ok":True }


@app .post ("/simulate")
def simulate_endpoint (payload :SimulateReq )->Dict [str ,Any ]:
# monta config do simulador
    cfg =SimCfg (
    sqlite_path =payload .sqlite_path ,
    trackId =payload .trackId ,
    race_laps =payload .race_laps ,
    setup_mode =payload .setup_mode ,
    style_code =payload .style_code ,
    allowed_compounds =payload .allowed_compounds ,
    two_compounds_rule =payload .two_compounds_rule ,
    pit_loss_s =payload .pit_loss_s ,
    max_stints =payload .max_stints ,
    max_stint_laps =payload .max_stint_laps ,
    min_stint_laps =payload .min_stint_laps ,
    use_cache_in_process =True ,
    )

    # empacota extras (sem quebrar seu simulate atual)
    out =simulate (cfg )

    # anexa parâmetros extras p/ o front não 'chutar'
    out ["resolved"]["min_compound_total_laps"]=int (payload .min_compound_total_laps )
    out ["resolved"]["drift_s_per_lap"]=float (payload .drift_s_per_lap )

    return out 