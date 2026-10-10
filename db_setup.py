#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database setup for Nexus Offline Assistant (called by setup_and_run.bat).

Two modes, auto-detected:
  A. MySQL/MariaDB already on this machine (port 3306/3307 answer)
     -> uses it: creates `chatbot_db`, imports schema.sql.
        Root password via --root-password or MYSQL_ROOT_PASSWORD env
        (or typed when prompted by the .bat).
  B. Nothing installed
     -> downloads portable MariaDB (~90 MB), installs it as a Windows
        service named NexusDB, sets a random root password, creates
        `chatbot_db`, imports schema.sql. Needs admin rights.

Always writes the working credentials to `.env` next to this script
(app.py loads it automatically).

Usage:
  python db_setup.py [--dry-run] [--root-password XXX] [--port N]
Exit codes: 0 ok, 2 needs-admin, 3 failed (message printed).
"""

import os
import re
import sys
import time
import socket
import shutil
import secrets
import subprocess
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE, '.env')
SCHEMA = os.path.join(BASE, 'schema.sql')
MARIADB_URL = ('https://archive.mariadb.org/mariadb-11.4.5/'
               'winx64-packages/mariadb-11.4.5-winx64.zip')
MARIADB_DIR = os.path.join(BASE, 'mariadb')
SERVICE_NAME = 'NexusDB'
DB_NAME = 'chatbot_db'


def log(msg):
    print(f'[db-setup] {msg}', flush=True)


def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(('127.0.0.1', port)) == 0
    finally:
        s.close()


def is_admin():
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def read_env():
    vals = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    vals[k.strip()] = v.strip().strip('"')
    return vals


def write_env(vals):
    merged = read_env()
    merged.update(vals)
    with open(ENV_FILE, 'w', encoding='utf-8') as f:
        for k, v in merged.items():
            f.write(f'{k}={v}\n')


def split_schema(path):
    with open(path, encoding='utf-8') as f:
        sql = f.read()
    sql = re.sub(r'--[^\n]*', '', sql)
    return [s.strip() for s in sql.split(';') if s.strip()]


def apply_schema(pymysql, conn):
    for stmt in split_schema(SCHEMA):
        with conn.cursor() as cur:
            cur.execute(stmt)
    conn.commit()


def import_schema(host, port, user, password):
    import pymysql
    conn = pymysql.connect(host=host, port=port, user=user, password=password,
                           charset='utf8mb4', autocommit=False,
                           connect_timeout=10)
    try:
        apply_schema(pymysql, conn)
    finally:
        conn.close()


def find_free_port():
    for p in (3306, 3307, 3308):
        if not port_open(p):
            return p
    raise RuntimeError('no free MySQL port (3306-3308 busy)')


def provision(root_password, port, dry_run=False):
    """Install portable MariaDB as service + import schema."""
    if dry_run:
        log(f'DRY-RUN: would download {MARIADB_URL}')
        log(f'DRY-RUN: would init service {SERVICE_NAME} on port {port}')
        return
    if not is_admin():
        log('NEEDS-ADMIN: right-click setup_and_run.bat -> Run as administrator')
        sys.exit(2)
    os.makedirs(MARIADB_DIR, exist_ok=True)
    zip_path = os.path.join(MARIADB_DIR, 'mariadb.zip')
    marker = os.path.join(MARIADB_DIR, 'bin', 'mysqld.exe')
    if not os.path.exists(marker):
        log('downloading portable MariaDB (~90 MB, resumes if broken)...')
        req = urllib.request.Request(MARIADB_URL,
                                     headers={'User-Agent': 'NexusSetup/1.0'})
        have = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        if have:
            req.add_header('Range', f'bytes={have}-')
        with urllib.request.urlopen(req, timeout=60) as r, \
                open(zip_path, 'ab') as f:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        log('extracting...')
        with zipfile.ZipFile(zip_path) as z:
            top = z.namelist()[0].split('/')[0]
            z.extractall(MARIADB_DIR)
        inner = os.path.join(MARIADB_DIR, top)
        for item in os.listdir(inner):
            shutil.move(os.path.join(inner, item), os.path.join(MARIADB_DIR, item))
        os.rmdir(inner)
        os.remove(zip_path)
    else:
        log('portable MariaDB already extracted.')

    data_dir = os.path.join(MARIADB_DIR, 'data')
    bin_dir = os.path.join(MARIADB_DIR, 'bin')
    mysqld = os.path.join(bin_dir, 'mysqld.exe')
    installer = os.path.join(bin_dir, 'mariadb-install-db.exe')
    mysqladmin = os.path.join(bin_dir, 'mysqladmin.exe')

    if not os.path.exists(os.path.join(data_dir, 'mysql')):
        log('initializing data directory...')
        subprocess.run([installer, f'--datadir={data_dir}',
                        f'--password={root_password}', '--skip-test-db'],
                       check=True, capture_output=True)
    # (re)install service on our port
    subprocess.run(['sc', 'delete', SERVICE_NAME], capture_output=True)
    subprocess.run([mysqld, '--install', SERVICE_NAME,
                    f'--datadir={data_dir}', f'--port={port}'],
                   check=True, capture_output=True)
    subprocess.run(['sc', 'start', SERVICE_NAME], check=True, capture_output=True)
    log(f'service {SERVICE_NAME} started on port {port}.')
    for _ in range(30):
        if port_open(port):
            break
        time.sleep(1)
    else:
        raise RuntimeError('database service did not start listening')
    # installer sets root password only for some auth paths; enforce it
    try:
        subprocess.run([mysqladmin, f'--port={port}', '-u', 'root',
                        'password', root_password],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pass  # password already set — fine
    import_schema('127.0.0.1', port, 'root', root_password)
    log('schema imported.')


def main():
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    root_pw = None
    port_arg = None
    for i, a in enumerate(args):
        if a == '--root-password' and i + 1 < len(args):
            root_pw = args[i + 1]
        if a == '--port' and i + 1 < len(args):
            port_arg = int(args[i + 1])
    if not root_pw:
        root_pw = os.getenv('MYSQL_ROOT_PASSWORD', '')

    if port_open(3306) or (port_arg and port_open(port_arg)):
        port = port_arg or 3306
        log(f'database server already present (port {port}) — reusing it.')
        if not root_pw:
            log('NEEDS-PASSWORD: set MYSQL_ROOT_PASSWORD env or pass --root-password, then re-run.')
            sys.exit(3)
        if dry_run:
            log('DRY-RUN: would import schema using provided password.')
            return
        import_schema('127.0.0.1', port, 'root', root_pw)
        write_env({'MYSQL_HOST': '127.0.0.1', 'MYSQL_PORT': str(port),
                   'MYSQL_USER': 'root', 'MYSQL_PASSWORD': root_pw,
                   'MYSQL_DB': DB_NAME})
        log('schema imported into existing server; credentials saved to .env')
        return

    port = port_arg or find_free_port()
    if not root_pw:
        root_pw = secrets.token_hex(12)
        log('generated a random root password (will be saved to .env).')
    provision(root_pw, port, dry_run=dry_run)
    if dry_run:
        return
    write_env({'MYSQL_HOST': '127.0.0.1', 'MYSQL_PORT': str(port),
               'MYSQL_USER': 'root', 'MYSQL_PASSWORD': root_pw,
               'MYSQL_DB': DB_NAME, 'MYSQL_SERVICE': SERVICE_NAME})
    log('done. credentials saved to .env')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f'FAILED: {e}')
        sys.exit(3)
