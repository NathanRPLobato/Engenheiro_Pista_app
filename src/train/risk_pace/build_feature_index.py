from __future__ import annotations 

import argparse 
from pathlib import Path 

import pandas as pd 
import numpy as np 


def main ()->None :
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--feature_dir",required =True )
    ap .add_argument ("--out_name",default ="feature_index.parquet")
    args =ap .parse_args ()

    feature_dir =Path (args .feature_dir )
    files =sorted ([p for p in feature_dir .glob ("*.parquet")if p .is_file ()])

    if not files :
        raise SystemExit (f"Nenhum parquet encontrado em: {feature_dir}")

    rows =[]
    for p in files :
    # lê só colunas mínimas p/ indexar (barato)
        cols =["trackId","race_id","lap_number","y_delta"]
        df =pd .read_parquet (p ,columns =[c for c in cols if c in pd .read_parquet (p ,engine ="pyarrow").columns ])

        if "trackId"not in df .columns or "race_id"not in df .columns :
        # se algum arquivo estiver 'bugado', você já descobre aqui
            continue 

        y =pd .to_numeric (df ["y_delta"],errors ="coerce")if "y_delta"in df .columns else pd .Series (dtype =float )
        lap =pd .to_numeric (df ["lap_number"],errors ="coerce")if "lap_number"in df .columns else pd .Series (dtype =float )

        rows .append ({
        "path":str (p ),
        "trackId":str (df ["trackId"].iloc [0 ]),
        "race_id":str (df ["race_id"].iloc [0 ]),
        "n_rows":int (len (df )),
        "lap_min":int (np .nanmin (lap ))if len (lap )else None ,
        "lap_max":int (np .nanmax (lap ))if len (lap )else None ,
        "y_mean":float (np .nanmean (y ))if len (y )else None ,
        "y_std":float (np .nanstd (y ))if len (y )else None ,
        })

    idx =pd .DataFrame (rows ).sort_values (["trackId","race_id"]).reset_index (drop =True )

    out_path =feature_dir /args .out_name 
    idx .to_parquet (out_path ,index =False )
    print (f"[ok] wrote index: {out_path} rows={len(idx)}")


if __name__ =="__main__":
    main ()