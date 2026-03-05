# arquivo: src/api/db.py
from __future__ import annotations 

import sqlite3 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Optional 


@dataclass (frozen =True )
class DBConfig :
    sqlite_path :str ="data/refined/telemetry.sqlite"
    timeout_s :float =5.0 


def connect_db (cfg :DBConfig )->sqlite3 .Connection :
    db_path =Path (cfg .sqlite_path )
    if not db_path .exists ():
        raise FileNotFoundError (f"SQLite não existe em: {db_path}")

    con =sqlite3 .connect (str (db_path ),timeout =cfg .timeout_s )

    # pragmas 'seguros' e bons pra leitura concorrente
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    con .execute ("PRAGMA temp_store=MEMORY;")
    con .execute ("PRAGMA foreign_keys=ON;")
    return con 