from __future__ import annotations
import datetime as dt
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

def utcnow(): return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

class Base(DeclarativeBase): pass

class Folder(Base):
    __tablename__ = "folders"
    __table_args__ = (UniqueConstraint("parent_id", "name", name="uq_folder_parent_name"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("folders.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    original_name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    mime: Mapped[str] = mapped_column(String(60), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    folder_id: Mapped[int | None] = mapped_column(ForeignKey("folders.id"), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(20), default="pending")        # pending | processing | completed | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(10), nullable=True)               # ACCEPT | REVIEW (final system decision)
    decision_reasons: Mapped[list] = mapped_column(JSON, default=list)
    review_status: Mapped[str] = mapped_column(String(20), default="unreviewed")          # unreviewed | approved | rejected | corrected
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    processing_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    ocr_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    ocr_mean_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    runs: Mapped[list["ExtractionRun"]] = relationship(back_populates="document", cascade="all, delete-orphan", order_by="ExtractionRun.attempt_no")
    fields: Mapped[list["FieldRecord"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    reviews: Mapped[list["ReviewEvent"]] = relationship(back_populates="document", cascade="all, delete-orphan", order_by="ReviewEvent.id")

class ExtractionRun(Base):
    __tablename__ = "extraction_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer)                                      # 0 = initial, 1.. = retries
    kind: Mapped[str] = mapped_column(String(10))                                         # initial | retry
    status: Mapped[str] = mapped_column(String(15))                                       # ok | failed | parse_failed
    model: Mapped[str] = mapped_column(String(80), default="")
    prompt_hash: Mapped[str] = mapped_column(String(20), default="")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage: Mapped[dict] = mapped_column(JSON, default=dict)
    latency_s: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)                # retry outcome: improved | no_improvement | regressed | failed
    validation: Mapped[list] = mapped_column(JSON, default=list)                          # all rule results of this attempt
    doc_decision: Mapped[str | None] = mapped_column(String(10), nullable=True)           # routing of THIS attempt (RETRY/REVIEW/ACCEPT)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)                        # True = this attempt's values are the current result
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    document: Mapped[Document] = relationship(back_populates="runs")

class FieldRecord(Base):
    __tablename__ = "field_records"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("extraction_runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(40), index=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)                        # as extracted
    norm_value: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)   # normalised (amounts: 12345.67) for search
    source_lines: Mapped[list] = mapped_column(JSON, default=list)
    ocr_signal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    llm_signal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    decision: Mapped[str] = mapped_column(String(10), default="ABSENT")
    issues: Mapped[list] = mapped_column(JSON, default=list)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)                      # belongs to the current result
    reviewed_value: Mapped[str | None] = mapped_column(Text, nullable=True)               # human edit
    was_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    document: Mapped[Document] = relationship(back_populates="fields")
    @property
    def final_value(self): return self.reviewed_value if self.was_edited else self.value

class ReviewEvent(Base):
    __tablename__ = "review_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    action: Mapped[str] = mapped_column(String(20))                                       # edited | approved | rejected | reprocessed | moved
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    changes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    document: Mapped[Document] = relationship(back_populates="reviews")
