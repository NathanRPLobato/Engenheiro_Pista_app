# arquivo: src/data/materialize_telemetry_flat.py
from __future__ import annotations 

import argparse 
import sqlite3 
import time 
from pathlib import Path 


DEFAULT_DB =Path ("data/refined/telemetry.sqlite")


def _exists (con :sqlite3 .Connection ,name :str )->bool :
    row =con .execute (
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
    (name ,),
    ).fetchone ()
    return row is not None 


def _cols (con :sqlite3 .Connection ,table :str )->list [str ]:
    rows =con .execute (f"PRAGMA table_info('{table}')").fetchall ()
    return [r [1 ]for r in rows ]


def _count (con :sqlite3 .Connection ,sql :str )->int :
    return int (con .execute (sql ).fetchone ()[0 ])


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--db",default =str (DEFAULT_DB ))
    ap .add_argument ("--drop_and_rebuild",action ="store_true")
    ap .add_argument ("--source_core",default ="telemetry_core")
    ap .add_argument ("--part1",default ="telemetry_part_001")
    ap .add_argument ("--part2",default ="telemetry_part_002")
    ap .add_argument ("--out_table",default ="telemetry_flat")
    ap .add_argument ("--create_view_telemetry",action ="store_true",help ="Cria VIEW telemetry apontando para telemetry_flat")
    args =ap .parse_args ()

    db_path =Path (args .db )
    if not db_path .exists ():
        raise SystemExit (f"SQLite não encontrado: {db_path}")

    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")

    if not _exists (con ,args .source_core ):
        raise SystemExit (f"Tabela não existe: {args.source_core}")
    if not _exists (con ,args .part1 ):
        raise SystemExit (f"Tabela não existe: {args.part1}")

    has_part2 =_exists (con ,args .part2 )

    core_cols =_cols (con ,args .source_core )
    p1_cols =_cols (con ,args .part1 )
    p2_cols =_cols (con ,args .part2 )if has_part2 else []

    # Checagens mínimas
    for needed in ["row_id","race_id","trackId","lap_number","lap_time"]:
        if needed not in core_cols :
            raise SystemExit (f"Coluna obrigatória ausente em {args.source_core}: {needed}")

            # validBin normalmente está na part_001, mas você falou: foda-se.
            # Mesmo assim eu vou materializar is_valid só pra deixar pronto.
    has_validBin ="validBin"in p1_cols 
    has_valid_bin_core ="valid_bin"in core_cols 

    # Monta lista de colunas da part1 e part2 sem row_id p/ não duplicar
    p1_sel =[c for c in p1_cols if c !="row_id"]
    p2_sel =[c for c in p2_cols if c !="row_id"]

    # Evita duplicar colunas que já existem no core (ex: trackLength etc)
    # Se colidir, renomeia com prefixo p1__/p2__
    core_set =set (core_cols )

    p1_proj =[]
    for c in p1_sel :
        if c in core_set :
            p1_proj .append (f"tp1.\"{c}\" AS \"p1__{c}\"")
        else :
            p1_proj .append (f"tp1.\"{c}\"")

    p2_proj =[]
    for c in p2_sel :
        if c in core_set or c in p1_sel :
            p2_proj .append (f"tp2.\"{c}\" AS \"p2__{c}\"")
        else :
            p2_proj .append (f"tp2.\"{c}\"")

            # is_valid calculado sem shadowing
    if has_validBin and has_valid_bin_core :
        valid_expr ="COALESCE(tp1.validBin, tc.valid_bin, 0)"
    elif has_validBin :
        valid_expr ="COALESCE(tp1.validBin, 0)"
    elif has_valid_bin_core :
        valid_expr ="COALESCE(tc.valid_bin, 0)"
    else :
        valid_expr ="0"

    if args .drop_and_rebuild :
        con .execute (f"DROP TABLE IF EXISTS {args.out_table};")
        con .execute ("DROP VIEW IF EXISTS telemetry;")
        con .execute ("DROP VIEW IF EXISTS telemetry_valid;")

    print ("\n=== MATERIALIZE TELEMETRY FLAT ===")
    print (f"db: {db_path}")
    print (f"core: {args.source_core}")
    print (f"part1: {args.part1}")
    print (f"part2: {args.part2}  has={has_part2}")
    print (f"out_table: {args.out_table}")
    print (f"is_valid expr: {valid_expr}")

    # Cria tabela materializada
    t0 =time .time ()

    select_list =[
    "tc.*",
    *p1_proj ,
    ]
    if has_part2 :
        select_list .extend (p2_proj )

    select_list .append (f"{valid_expr} AS is_valid")

    sql =f"""
    CREATE TABLE {args.out_table} AS
    SELECT
        {", ".join(select_list)}
    FROM {args.source_core} tc
    JOIN {args.part1} tp1
        ON tp1.row_id = tc.row_id
    {"LEFT JOIN " + args.part2 + " tp2 ON tp2.row_id = tc.row_id" if has_part2 else ""}
    WHERE tc.lap_number >= 0
    ;
    """

    con .execute (sql )
    con .commit ()
    t1 =time .time ()

    n_flat =_count (con ,f"SELECT COUNT(*) FROM {args.out_table};")
    print (f"[ok] telemetry_flat criado. linhas: {n_flat}  em {t1 - t0:.2f}s")

    # Índices que importam (você vai agradecer depois)
    print ("[index] criando índices...")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__race_lap ON {args.out_table}(race_id, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__race_track ON {args.out_table}(race_id, trackId);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__track_lap ON {args.out_table}(trackId, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.out_table}__is_valid ON {args.out_table}(is_valid);")
    con .commit ()
    print ("[ok] índices criados")

    # Opcional: cria view telemetry apontando pro flat
    if args .create_view_telemetry :
        con .execute ("DROP VIEW IF EXISTS telemetry;")
        con .execute (f"CREATE VIEW telemetry AS SELECT * FROM {args.out_table};")
        con .execute ("DROP VIEW IF EXISTS telemetry_valid;")
        con .execute ("CREATE VIEW telemetry_valid AS SELECT * FROM telemetry WHERE is_valid = 1;")
        con .commit ()
        n_all =_count (con ,"SELECT COUNT(*) FROM telemetry;")
        n_valid =_count (con ,"SELECT COUNT(*) FROM telemetry_valid;")
        print (f"[ok] view telemetry criada. linhas: {n_all}")
        print (f"[ok] view telemetry_valid criada. linhas: {n_valid}")

    con .close ()
    print ("\nPronto.")


if __name__ =="__main__":
    main ()
