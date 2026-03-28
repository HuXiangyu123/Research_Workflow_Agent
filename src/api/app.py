from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agent.llm import build_deepseek_chat
from src.agent.react_agent import build_react_agent
from src.agent.report import generate_literature_report
from src.agent.settings import Settings
from src.tools.pdf import extract_text_from_pdf_bytes

load_dotenv()

PDF_SUFFIX = ".pdf"

app = FastAPI(title="Literature Report Agent", version="0.1.0")


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
