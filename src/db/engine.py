"""数据库引擎与会话管理（PostgreSQL + SQLAlchemy 2.0）。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# 向量扩展（可选，缺失时不报错）
try:
    import pgvector
    from pgvector.sqlalchemy import Vector  # type: ignore[import]
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    Vector = object  # 仅为类型标注，向量字段可降级为 ARRAY(float)

Base = declarative_base()


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Please configure it in .env (e.g. postgresql://researchuser:123@127.0.0.1:5432/researchagent)"
        )
    return url


def build_engine(echo: bool = False) -> Engine:
    url = _get_database_url()
    engine = create_engine(
        url,
        echo=echo,
        poolclass=NullPool,
        connect_args={"options": "-c search_path=public"},
    )
    if _HAS_PGVECTOR:
        _register_vector_extension(engine)
    return engine


def _register_vector_extension(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        try:
            with dbapi_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            pass  # 生产环境可能无 superuser，降级处理


_ENGINE: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = build_engine()
    return _ENGINE


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def reset_engine() -> None:
    """Dispose cached SQLAlchemy engine/session factory.

    Tests and local config reloads use this after changing DATABASE_URL or
    persistence settings.
    """
    global _ENGINE, _SessionFactory
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SessionFactory = None


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """创建所有表（首次运行调用）。"""
    from src.db.models import Chunk, Document  # noqa: F401
    engine = get_engine()
    # checkfirst=True 避免重复 create table，但索引重复仍可能报错，异常时忽略
    try:
        Base.metadata.create_all(engine, checkfirst=True)
    except Exception:
        pass


def health_check() -> bool:
    """验证数据库连通性。"""
    try:
        with get_db_session() as s:
            s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
