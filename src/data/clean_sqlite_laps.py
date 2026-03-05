# arquivo: src/data/clean_sqlite_laps.py
from __future__ import annotations 

import argparse 
import sqlite3 
import time 
from pathlib import Path 

import pandas as pd 

SQLITE_PATH =Path ("data/refined/telemetry.sqlite")


def _table_exists (con :sqlite3 .Connection ,table :str )->bool :
    cur =con .execute (
    "SELECT 1 FROM sqlite_master WHERE (type='table' OR type='view') AND name=?",
    (table ,),
    )
    return cur .fetchone ()is not None 


def _has_col (con :sqlite3 .Connection ,table :str ,col :str )->bool :
    rows =con .execute (f"PRAGMA table_info('{table}')").fetchall ()
    return any (r [1 ]==col for r in rows )


def _scalar_int (con :sqlite3 .Connection ,sql :str )->int :
    return int (con .execute (sql ).fetchone ()[0 ])


def _scalar_float (con :sqlite3 .Connection ,sql :str )->float :
    v =con .execute (sql ).fetchone ()[0 ]
    return float (v )if v is not None else float ("nan")


def _print_binindex_diagnostics (con :sqlite3 .Connection ,table :str ):
# Mostra se binIndex está bugado (NULL, -1, constante, etc)
    print ("\n[diag] binIndex quick check:")
    try :
        total =_scalar_int (con ,f"SELECT COUNT(*) FROM {table};")
        nulls =_scalar_int (con ,f"SELECT COUNT(*) FROM {table} WHERE binIndex IS NULL;")
        neg1 =_scalar_int (con ,f"SELECT COUNT(*) FROM {table} WHERE binIndex = -1;")
        mn =con .execute (f"SELECT MIN(binIndex) FROM {table} WHERE binIndex IS NOT NULL;").fetchone ()[0 ]
        mx =con .execute (f"SELECT MAX(binIndex) FROM {table} WHERE binIndex IS NOT NULL;").fetchone ()[0 ]
        print (f"  total={total}  binIndex_null={nulls}  binIndex_-1={neg1}  min={mn}  max={mx}")

        # Top valores mais frequentes, pra pegar caso 'constante'
        top =pd .read_sql (
        f"""
            SELECT binIndex, COUNT(*) AS n
            FROM {table}
            GROUP BY binIndex
            ORDER BY n DESC
            LIMIT 10;
            """,
        con ,
        )
        print (top )
    except Exception as e :
        print (f"  [diag] falhou: {e}")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sqlite_path",default =str (SQLITE_PATH ))
    ap .add_argument ("--table",default ="telemetry_core")
    ap .add_argument ("--dry_run",action ="store_true")
    ap .add_argument ("--force",action ="store_true",help ="Ignora o freio de segurança (use com cuidado)")
    ap .add_argument ("--delete_invalid_lap_number",action ="store_true",help ="Remove lap_number < 0")
    ap .add_argument ("--delete_invalid_lap_time",action ="store_true",help ="Remove lap_time <= 0 (se existir)")
    ap .add_argument ("--dedup_binindex",action ="store_true",help ="Dedup por (race_id, binIndex) mantendo a última amostra")
    ap .add_argument ("--vacuum",action ="store_true",help ="Roda VACUUM no final (demora)")
    args =ap .parse_args ()

    db_path =Path (args .sqlite_path )
    if not db_path .exists ():
        raise SystemExit (f"SQLite não encontrado: {db_path}")

    con =sqlite3 .connect (db_path )
    con .execute ("PRAGMA journal_mode=WAL;")
    con .execute ("PRAGMA synchronous=NORMAL;")
    con .execute ("PRAGMA temp_store=MEMORY;")

    if not _table_exists (con ,args .table ):
        con .close ()
        raise SystemExit (f"Tabela '{args.table}' não existe nesse SQLite.")

        # Confere colunas básicas
    if not _has_col (con ,args .table ,"race_id"):
        con .close ()
        raise SystemExit ("Coluna obrigatória ausente: race_id")
    if not _has_col (con ,args .table ,"row_id"):
        con .close ()
        raise SystemExit ("Coluna obrigatória ausente: row_id (PK)")

    total =_scalar_int (con ,f"SELECT COUNT(*) FROM {args.table};")

    # Diagnóstico do binIndex (só se for usar dedup)
    if args .dedup_binindex and _has_col (con ,args .table ,"binIndex"):
        _print_binindex_diagnostics (con ,args .table )

        # Contagens de candidatos a delete
    to_drop_lap =0 
    if args .delete_invalid_lap_number and _has_col (con ,args .table ,"lap_number"):
        to_drop_lap =_scalar_int (con ,f"SELECT COUNT(*) FROM {args.table} WHERE lap_number < 0;")

    to_drop_time =0 
    if args .delete_invalid_lap_time and _has_col (con ,args .table ,"lap_time"):
        to_drop_time =_scalar_int (con ,f"SELECT COUNT(*) FROM {args.table} WHERE lap_time <= 0;")

        # Duplicatas 'excesso': (total_válido) - (qtd_grupos_distintos)
    dup_excess =0 
    dedup_scope_total =0 

    if args .dedup_binindex :
        if not _has_col (con ,args .table ,"binIndex"):
            con .close ()
            raise SystemExit ("Você pediu --dedup_binindex mas a coluna binIndex não existe.")

            # IMPORTANTÍSSIMO:
            # Só deduplica onde binIndex é realmente identificador: NOT NULL e = -1.
        dedup_scope_total =_scalar_int (
        con ,
        f"""
            SELECT COUNT(*)
            FROM {args.table}
            WHERE binIndex IS NOT NULL
              AND binIndex != -1;
            """,
        )
        distinct_groups =_scalar_int (
        con ,
        f"""
            SELECT COUNT(*)
            FROM (
              SELECT race_id, binIndex
              FROM {args.table}
              WHERE binIndex IS NOT NULL
                AND binIndex != -1
              GROUP BY race_id, binIndex
            ) g;
            """,
        )
        dup_excess =max (0 ,dedup_scope_total -distinct_groups )

    would_delete =to_drop_lap +to_drop_time +dup_excess 
    pct =(100.0 *would_delete /total )if total >0 else 0.0 

    print ("\n=== CLEAN SQLITE (telemetry samples) ===")
    print (f"db: {db_path}")
    print (f"table: {args.table}")
    print (f"rows_total: {total}")
    if args .delete_invalid_lap_number :
        print (f"rows_lap_number<0: {to_drop_lap}")
    if args .delete_invalid_lap_time :
        print (f"rows_lap_time<=0: {to_drop_time}")
    if args .dedup_binindex :
        print (f"dedup_scope_total (binIndex válido): {dedup_scope_total}")
        print (f"duplicatas_excesso (race_id, binIndex): {dup_excess}")
    print (f"would_delete: {would_delete} ({pct:.2f}%)")

    if args .dry_run :
        print ("[dry_run] nada foi deletado.")
        con .close ()
        return 

        # Freio de segurança: se tentar deletar mais que 60% sem querer
    if (pct >60.0 )and (not args .force ):
        con .close ()
        raise SystemExit (
        f"[ABORTADO] Isso deletaria {pct:.2f}% da base. "
        "Provável coluna errada ou binIndex bugado (NULL/-1/constante). "
        "Se você tem certeza, rode com --force."
        )

    t0 =time .time ()

    # 1) lap_number < 0
    if args .delete_invalid_lap_number and _has_col (con ,args .table ,"lap_number"):
        con .execute (f"DELETE FROM {args.table} WHERE lap_number < 0;")

        # 2) lap_time <= 0
    if args .delete_invalid_lap_time and _has_col (con ,args .table ,"lap_time"):
        con .execute (f"DELETE FROM {args.table} WHERE lap_time <= 0;")

        # 3) dedup mantendo a última amostra (maior row_id) por (race_id, binIndex)
    if args .dedup_binindex :
    # Deleta somente no escopo binIndex válido (NOT NULL e = -1)
    # Mantém o registro com maior row_id em cada grupo.
        con .execute (
        f"""
            DELETE FROM {args.table}
            WHERE row_id IN (
              SELECT row_id
              FROM {args.table}
              WHERE binIndex IS NOT NULL
                AND binIndex != -1
              EXCEPT
              SELECT MAX(row_id)
              FROM {args.table}
              WHERE binIndex IS NOT NULL
                AND binIndex != -1
              GROUP BY race_id, binIndex
            );
            """
        )

    con .commit ()
    t1 =time .time ()

    after =_scalar_int (con ,f"SELECT COUNT(*) FROM {args.table};")
    deleted =total -after 

    print (f"\n[ok] deletado: {deleted} linhas")
    print (f"[ok] rows_after: {after}")
    print (f"[ok] time_seconds: {t1 - t0:.2f}s")

    # Índices úteis
    print ("\n[index] criando índices...")
    t2 =time .time ()
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.table}__race_lap ON {args.table}(race_id, lap_number);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.table}__race_track ON {args.table}(race_id, trackId);")
    con .execute (f"CREATE INDEX IF NOT EXISTS idx_{args.table}__race_bin ON {args.table}(race_id, binIndex);")
    con .commit ()
    t3 =time .time ()
    print (f"[ok] índices em {t3 - t2:.2f}s")

    if args .vacuum :
        print ("\n[vacuum] checkpoint + vacuum (pode demorar)...")
        con .execute ("PRAGMA wal_checkpoint(TRUNCATE);")
        con .commit ()
        con .execute ("VACUUM;")
        con .commit ()
        print ("[ok] vacuum concluído")

    con .close ()


if __name__ =="__main__":
    main ()
