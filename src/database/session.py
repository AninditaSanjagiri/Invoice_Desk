from __future__ import annotations
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from .models import Base

def make_engine(db_path):
    url = "sqlite://" if str(db_path) == ":memory:" else f"sqlite:///{db_path}"
    kw = {"connect_args": {"check_same_thread": False}}
    if url == "sqlite://":
        from sqlalchemy.pool import StaticPool; kw["poolclass"] = StaticPool
    eng = create_engine(url, **kw)
    @event.listens_for(eng, "connect")
    def _fk(dbapi_con, _):                   # enforce foreign keys in SQLite
        dbapi_con.execute("PRAGMA foreign_keys=ON")
    return eng

def init_db(db_path):
    """Creates all tables if missing (idempotent). Returns a session factory."""
    eng = make_engine(db_path); Base.metadata.create_all(eng)
    return sessionmaker(eng, expire_on_commit=False)

@contextmanager
def session_scope(factory):
    s = factory()
    try: yield s; s.commit()
    except Exception: s.rollback(); raise
    finally: s.close()
