from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.llm import build_deepseek_chat
from src.agent.react_agent import build_react_agent
from src.agent.report import generate_literature_report
from src.agent.settings import Settings
from src.api.routes.agents import router as agents_router
from src.api.routes.config import router as config_router
from src.api.routes.mcp import router as mcp_router
from src.api.routes.skills import router as skills_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.corpus_evidence import router as corpus_evidence_router
from src.api.routes.evals import router as evals_router
from src.api.routes.workspaces import router as workspaces_router
from src.tools.pdf import extract_text_from_pdf_bytes

load_dotenv(".env")

PDF_SUFFIX = ".pdf"

app = FastAPI(title="Literature Report Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(tasks_router)
app.include_router(corpus_evidence_router)
app.include_router(evals_router)
app.include_router(agents_router)
app.include_router(config_router)
app.include_router(skills_router)
app.include_router(mcp_router)
app.include_router(workspaces_router)


def _build_react_agent():
    settings = Settings.from_env()
    llm = build_deepseek_chat(settings)
    return build_react_agent(llm)


class ReportRequest(BaseModel):
    arxiv_url_or_id: str = Field(..., description="arXiv 链接或 arXiv ID")


class ReportResponse(BaseModel):
    report: str


@app.post("/report", response_model=ReportResponse)
def generate_report(payload: ReportRequest) -> ReportResponse:
    try:
        agent = _build_react_agent()
        report = generate_literature_report(agent, arxiv_url_or_id=payload.arxiv_url_or_id)
        return ReportResponse(report=report)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/report/upload_pdf", response_model=ReportResponse)
async def generate_report_from_pdf(file: UploadFile = File(...)) -> ReportResponse:
    """上传 PDF 文件，解析内容并生成文献报告。"""
    name = (file.filename or "").lower()
    if not name.endswith(PDF_SUFFIX):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf_bytes(content)

        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF (maybe scanned?)")

        agent = _build_react_agent()
        report = generate_literature_report(agent, raw_text_content=pdf_text)
        return ReportResponse(report=report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
