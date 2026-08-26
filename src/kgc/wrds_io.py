"""WRDS 存取層。

所有取數走 `extract()`：它執行 SQL、把結果落到 `data/` 底下的 parquet，
並在旁邊寫一份 manifest 記錄 SQL、參數、列數、SHA-256、抽取時間與 git commit。
原始資料受 WRDS 授權限制不進版控，manifest 才是可提交、可重現的憑證。

連線密碼一律從 `~/.pgpass` 讀，不接受以參數或環境變數傳入。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ENV_PATH = REPO_ROOT / ".env"
PGPASS_PATH = Path.home() / ".pgpass"

DEFAULTS = {
    "WRDS_HOST": "wrds-pgdata.wharton.upenn.edu",
    "WRDS_PORT": "9737",
    "WRDS_DBNAME": "wrds",
}


class PgpassMissingError(RuntimeError):
    """~/.pgpass 不存在或權限不對，libpq 會安靜地忽略它。"""


def load_env(path: Path = ENV_PATH) -> None:
    """讀取 .env，不覆蓋已存在的環境變數。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _check_pgpass() -> None:
    if not PGPASS_PATH.exists():
        raise PgpassMissingError(
            f"{PGPASS_PATH} 不存在。建立方式見 README。"
        )
    mode = PGPASS_PATH.stat().st_mode & 0o077
    if mode:
        raise PgpassMissingError(
            f"{PGPASS_PATH} 權限過寬（group/other 可讀），libpq 會忽略它。"
            " 執行 chmod 400 ~/.pgpass"
        )


def connect(readonly: bool = True):
    """開一條 WRDS 連線。密碼由 libpq 從 ~/.pgpass 取得。"""
    load_env()
    _check_pgpass()

    user = os.environ.get("WRDS_USER")
    if not user:
        raise RuntimeError("WRDS_USER 未設定，請在 .env 填入。")

    conn = psycopg2.connect(
        host=os.environ.get("WRDS_HOST", DEFAULTS["WRDS_HOST"]),
        port=int(os.environ.get("WRDS_PORT", DEFAULTS["WRDS_PORT"])),
        dbname=os.environ.get("WRDS_DBNAME", DEFAULTS["WRDS_DBNAME"]),
        user=user,
        sslmode="require",
        connect_timeout=20,
    )
    conn.set_session(readonly=readonly, autocommit=True)
    return conn


def _git_commit() -> str | None:
    """None 代表尚無 commit（新 repo）或不在 git 底下。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    name: str
    sql: str
    params: list | None
    rows: int
    columns: list[str]
    output: str
    sha256: str
    extracted_at: str
    git_commit: str | None
    wrds_user: str
    note: str | None = None


def extract(
    name: str,
    sql: str,
    params: tuple | list | None = None,
    *,
    conn=None,
    subdir: str = "",
    statement_timeout_ms: int = 600_000,
    note: str | None = None,
) -> tuple[pd.DataFrame, Manifest]:
    """執行 SQL，落檔到 data/<subdir>/<name>.parquet，並寫出 manifest。

    回傳 (DataFrame, Manifest)。呼叫端不需要自己處理落檔、hash 或 provenance。
    """
    owns_conn = conn is None
    if owns_conn:
        conn = connect()

    try:
        with conn.cursor() as cur:
            cur.execute(f"set statement_timeout = {statement_timeout_ms}")
            cur.execute(sql, params)
            columns = [d.name for d in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        if owns_conn:
            conn.close()

    out_dir = DATA_DIR / subdir if subdir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)

    manifest = Manifest(
        name=name,
        sql=" ".join(sql.split()),
        params=list(params) if params else None,
        rows=len(df),
        columns=list(df.columns),
        output=str(out_path.relative_to(REPO_ROOT)),
        sha256=_sha256(out_path),
        extracted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        wrds_user=os.environ.get("WRDS_USER", "?"),
        note=note,
    )
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df, manifest


def load(name: str, subdir: str = "") -> pd.DataFrame:
    """讀回先前 extract 的結果，並驗證 hash 與 manifest 相符。"""
    out_dir = DATA_DIR / subdir if subdir else DATA_DIR
    path = out_dir / f"{name}.parquet"
    manifest_path = out_dir / f"{name}.manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先跑對應的 extract 腳本。")

    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        actual = _sha256(path)
        if recorded != actual:
            raise RuntimeError(
                f"{path} 的 SHA-256 與 manifest 不符（資料被改動過）。"
                f"\n  manifest: {recorded}\n  actual:   {actual}"
            )
    return pd.read_parquet(path)
