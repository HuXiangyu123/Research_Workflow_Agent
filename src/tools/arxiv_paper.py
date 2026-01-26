import re
import time
import urllib.parse
import feedparser
from langchain_core.tools import tool

def _extract_arxiv_id(url_or_id: str) -> str | None:
    """
    从输入字符串中提取 arXiv ID。
    支持格式：
    - https://arxiv.org/abs/1706.03762
    - https://arxiv.org/pdf/1706.03762.pdf
    - 1706.03762
    - 1706.03762v5
    """
    # 移除可能的空白字符
    s = url_or_id.strip()
    
    # 匹配常见 arXiv ID 格式 (例如 1706.03762 或 1706.03762v1)
    # 简单的正则：\d{4}\.\d{4,5}(v\d+)?
    match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', s)
    if match:
        return match.group(1)
    
    return None

@tool
def get_arxiv_paper_info(arxiv_url_or_id: str) -> str:
    """
    通过 arXiv API 获取论文的元数据信息（标题、摘要、作者、发布时间等）。
    输入必须是 arXiv 论文链接（如 https://arxiv.org/abs/1706.03762）或 arXiv ID。
    """
    arxiv_id = _extract_arxiv_id(arxiv_url_or_id)
    if not arxiv_id:
        return f"Error: 无法从输入 '{arxiv_url_or_id}' 中解析出有效的 arXiv ID。"

    base_url = "http://export.arxiv.org/api/query"
    
    # 构建查询参数
    # search_query=id:xxxx
    params = {
        "search_query": f"id:{arxiv_id}",
        "start": 0,
        "max_results": 1
    }
    
    query_string = urllib.parse.urlencode(params)
    api_url = f"{base_url}?{query_string}"
    
    try:
        # 官方建议：如有连续请求需加延迟，这里单个请求暂时不需要，但保留 sleep 占位
        # time.sleep(3) 
        
        feed = feedparser.parse(api_url)
        
        if not feed.entries:
            return f"Error: 未在 arXiv 上找到 ID 为 {arxiv_id} 的论文。"

        entry = feed.entries[0]
        
        # 提取关键信息
        title = entry.title.replace('\n', ' ').strip()
        summary = entry.summary.replace('\n', ' ').strip()
        published = entry.published
        authors = ", ".join([author.name for author in entry.authors])
        
        # 查找 PDF 链接
        pdf_url = "N/A"
        # feedparser 解析出的 links 是字典或者对象，根据版本不同
        # 即使 entry.links 是列表，里面的元素也是 FeedParserDict
        if hasattr(entry, 'links'):
             for link in entry.links:
                # 检查 link 是否有 type 属性，并且是 pdf
                if hasattr(link, 'type') and link.type == 'application/pdf':
                     pdf_url = link.href
                # 或者通过 title 判断
                elif hasattr(link, 'title') and link.title == 'pdf':
                     pdf_url = link.href

        # 格式化输出
        result = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Published: {published}\n"
            f"PDF URL: {pdf_url}\n"
            f"Summary: {summary}\n"
        )
        return result

    except Exception as e:
        return f"Error fetching arXiv data: {str(e)}"
