# arquivo: src/data/ingest_raw_to_sqlite.py
from __future__ import annotations 

import argparse 
import re 
import shutil 
import sqlite3 
from pathlib import Path 
from typing import List ,Tuple 

import numpy as np 
import pandas as pd 

RAW_DIR =Path ("data/raw")
DONE_DIR =RAW_DIR /"_done"
SQLITE_PATH =Path ("data/refined/telemetry.sqlite")

DEFAULT_CHUNKSIZE =50_000 

# Limites práticos
# SQLite costuma ter limite 999 variáveis por statement.
# Se inserirmos por linha, variáveis = num_cols. Então precisamos num_cols < 999.
# Vamos usar margem.
MAX_COLS_PER_TABLE =250 # conservador e rápido, evita dor de cabeça


def _detect_delimiter (header_line :str )->str :
    candidates =["\t",",",";","|"]
    counts ={c :header_line .count (c )for c in candidates }
    delim =max (counts ,key =counts .get )
    return delim if counts [delim ]>0 else ","


def _read_header_first_line (path :Path )->str :
    with path .open ("r",encoding ="utf-8",errors ="ignore")as f :
        return f .readline ().strip ("\n\r")


def _normalize_columns (cols :List [str ])->List [str ]:
    out =[]
    for c in cols :
        c2 =c .strip ()
        c2 =re .sub (r"\s+","_",c2 )
        c2 =c2 .replace ("\ufeff","")
        out .append (c2 )

        # dedup
    seen ={}
    fixed =[]
    for c in out :
        if c not in seen :
            seen [c ]=0 
            fixed .append (c )
        else :
            seen [c ]+=1 
            fixed .append (f"{c}__dup{seen[c]}")
    return fixed 


def _infer_race_id_from_filename (path :Path )->str :
    return path .stem 


def _ensure_dirs ():
    SQLITE_PATH .parent .mkdir (parents =True ,exist_ok =True )
    DONE_DIR .mkdir (parents =True ,exist_ok =True )


def _recreate_sqlite (db_path :Path ):
    if db_path .exists ():
        db_path .unlink ()

    con =sqlite3 .connect (db_path )
    cur =con .cursor ()

    cur .execute (
    """
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            trackId INTEGER,
            trackLength REAL,
            weather INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )

    # Tabela core: vai ter row_id pra amarrar todo mundo
    cur .execute (
    """
        CREATE TABLE IF NOT EXISTS telemetry_core (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            valid_bin INTEGER,
            trackId INTEGER,
            lap_number INTEGER,
            lap_time REAL,
            binIndex INTEGER
        );
        """
    )

    cur .execute ("CREATE INDEX IF NOT EXISTS idx_core_race ON telemetry_core(race_id);")
    cur .execute ("CREATE INDEX IF NOT EXISTS idx_core_race_lap ON telemetry_core(race_id, lap_number);")
    cur .execute ("CREATE INDEX IF NOT EXISTS idx_core_race_lap_bin ON telemetry_core(race_id, lap_number, binIndex);")
    cur .execute ("CREATE INDEX IF NOT EXISTS idx_core_valid ON telemetry_core(valid_bin);")

    con .commit ()
    con .close ()


def _upsert_race_metadata (con :sqlite3 .Connection ,race_id :str ,df_head :pd .DataFrame ):
    def _first_valid (col :str ):
        if col not in df_head .columns :
            return None 
        v =pd .to_numeric (df_head [col ],errors ="coerce")
        v =v [np .isfinite (v )]
        return None if v .empty else float (v .iloc [0 ])

    track_id =_first_valid ("trackId")
    track_len =_first_valid ("trackLength")
    weather =_first_valid ("weather")

    con .execute (
    """
        INSERT INTO races (race_id, trackId, trackLength, weather)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(race_id) DO UPDATE SET
            trackId=excluded.trackId,
            trackLength=excluded.trackLength,
            weather=excluded.weather
        """,
    (
    race_id ,
    int (track_id )if track_id is not None else None ,
    float (track_len )if track_len is not None else None ,
    int (weather )if weather is not None else None ,
    ),
    )


def _pick_core_columns (all_cols :List [str ])->List [str ]:
    """
    Colunas que você sempre vai precisar pra:
    - baselines por volta
    - filtrar valid_bin
    - reconstruir stint e tempo final
    """
    preferred =["valid_bin","trackId","lap_number","lap_time","binIndex"]
    return [c for c in preferred if c in all_cols ]


def _split_columns (cols :List [str ],max_cols :int )->List [List [str ]]:
    out =[]
    for i in range (0 ,len (cols ),max_cols ):
        out .append (cols [i :i +max_cols ])
    return out 


def _create_part_table_if_needed (con :sqlite3 .Connection ,table :str ,cols :List [str ]):
    """
    Cada tabela part tem:
      row_id (PK, referencia telemetry_core.row_id)
      e as colunas da fatia
    """
    col_defs =[]
    for c in cols :
    # SQLite vai aceitar sem tipo fixo, mas vamos colocar REAL por padrão
    # Pra string também funciona, SQLite é dinâmico
        col_defs .append (f'"{c}"')

    cols_sql =", ".join ([f'"{c}"'for c in cols ])

    con .execute (
    f"""
        CREATE TABLE IF NOT EXISTS {table} (
            row_id INTEGER PRIMARY KEY,
            {", ".join([f'"{c}"' for c in cols])}
        );
        """
    )
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{table}_row ON {table}(row_id);")


def _insert_core (con :sqlite3 .Connection ,df_core :pd .DataFrame )->Tuple [int ,int ]:
    """
    Insere no core e retorna (first_row_id, n_rows).

    Observação importante:
    - Não usamos cursor.lastrowid depois de executemany, porque pode vir None.
    - Em vez disso, pegamos o MAX(row_id) antes e calculamos o intervalo.
    """
    cols =["race_id","valid_bin","trackId","lap_number","lap_time","binIndex"]

    # garante todas as colunas
    for c in cols :
        if c not in df_core .columns :
            df_core [c ]=None 

            # row_id que existe antes de inserir este chunk
    cur =con .cursor ()
    cur .execute ("SELECT COALESCE(MAX(row_id), 0) FROM telemetry_core;")
    prev_max_id =int (cur .fetchone ()[0 ])

    rows =list (df_core [cols ].itertuples (index =False ,name =None ))
    n =len (rows )
    if n ==0 :
        return prev_max_id +1 ,0 

    cur .executemany (
    """
        INSERT INTO telemetry_core (race_id, valid_bin, trackId, lap_number, lap_time, binIndex)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
    rows ,
    )

    first_id =prev_max_id +1 
    return first_id ,n 


def _insert_part (con :sqlite3 .Connection ,table :str ,row_ids :np .ndarray ,df_part :pd .DataFrame ):
    """
    Insere uma fatia de colunas para cada row_id.
    """
    cols =list (df_part .columns )
    _create_part_table_if_needed (con ,table ,cols )

    # Monta rows: (row_id, c1, c2, ...)
    # Importante: sem DataFrame.insert repetido pra não fragmentar
    values =df_part .to_numpy ()
    payload =[(int (rid ),*values [i ].tolist ())for i ,rid in enumerate (row_ids )]

    placeholders =", ".join (["?"]*(1 +len (cols )))
    colnames =", ".join (['"row_id"']+[f'"{c}"'for c in cols ])

    con .executemany (
    f"INSERT OR REPLACE INTO {table} ({colnames}) VALUES ({placeholders})",
    payload ,
    )


def ingest_one_file (
path :Path ,
db_path :Path ,
chunksize :int ,
move_done :bool =True ,
):
    race_id =_infer_race_id_from_filename (path )
    header_line =_read_header_first_line (path )
    sep =_detect_delimiter (header_line )

    print ("\n=== INGEST (RAW -> SQLite) ===")
    print (f"file={path.name}")
    print (f"race_id={race_id}")
    print (f"delimiter={repr(sep)}")
    print ("engine='c'  chunksize=%d"%chunksize )

    reader =pd .read_csv (
    path ,
    sep =sep ,
    engine ="c",
    chunksize =chunksize ,
    low_memory =False ,
    )

    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")

    first_chunk =True 
    total_in =0 
    total_written =0 

    for chunk in reader :
        total_in +=len (chunk )

        chunk .columns =_normalize_columns (list (chunk .columns ))

        # adiciona race_id sem fragmentar
        chunk =chunk .assign (race_id =race_id )

        if first_chunk :
            _upsert_race_metadata (con ,race_id ,chunk .head (2000 ))
            first_chunk =False 

            # Define core
        core_cols =_pick_core_columns (list (chunk .columns ))
        df_core =chunk [["race_id"]+core_cols ].copy ()

        # Insere core e pega row_ids desse bloco
        first_id ,n =_insert_core (con ,df_core )
        row_ids =np .arange (first_id ,first_id +n ,dtype =np .int64 )

        # Define colunas extras (tudo menos race_id e core)
        extra_cols =[c for c in chunk .columns if c not in (["race_id"]+core_cols )]

        # Split vertical em partes pequenas
        parts =_split_columns (extra_cols ,MAX_COLS_PER_TABLE )

        for pi ,cols in enumerate (parts ,start =1 ):
            df_part =chunk [cols ].copy ()
            table =f"telemetry_part_{pi:03d}"
            _insert_part (con ,table ,row_ids =row_ids ,df_part =df_part )

        total_written +=n 

        # commit por chunk pra não perder tudo se der pau no meio
        con .commit ()

    con .close ()

    print (f"[done] rows_in={total_in} rows_written={total_written}")
    print ("[done] tables: telemetry_core + telemetry_part_###")

    if move_done :
        dst =DONE_DIR /path .name 
        shutil .move (str (path ),str (dst ))
        print (f"[done] moved to: {dst}")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--raw_dir",default =str (RAW_DIR ))
    ap .add_argument ("--sqlite_path",default =str (SQLITE_PATH ))
    ap .add_argument ("--chunksize",type =int ,default =DEFAULT_CHUNKSIZE )
    ap .add_argument ("--no_move_done",action ="store_true")
    args =ap .parse_args ()

    raw_dir =Path (args .raw_dir )
    db_path =Path (args .sqlite_path )

    _ensure_dirs ()
    _recreate_sqlite (db_path )

    files =sorted (raw_dir .glob ("*.csv"))
    if not files :
        raise SystemExit (f"Nenhum .csv em {raw_dir}")

    for f in files :
        ingest_one_file (
        f ,
        db_path =db_path ,
        chunksize =args .chunksize ,
        move_done =(not args .no_move_done ),
        )

    print (f"\nSQLite criado em: {db_path}")
    print ("Tabela: races")
    print ("Tabela: telemetry_core")
    print ("Tabelas: telemetry_part_001, telemetry_part_002, ...")


if __name__ =="__main__":
    main ()
