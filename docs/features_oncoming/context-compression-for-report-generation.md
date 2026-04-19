# Context Compression for Report Generation Pipeline

> 生成时间：2026-04-14
> 状态：**Phase 1-3 已实现**（见下文）
> 优先级：P0（直接改善报告质量的核心问题）

---

## 实现状态

| 阶段 | 内容 | 状态 | 文件 |
|------|------|------|------|
| Phase 1 | `Taxonomy` + `CompressedCard` + `EvidencePool` 模型 | **✅ 完成** | `src/models/compression.py` |
| Phase 2 | 压缩服务核心实现（`_build_taxonomy` + `_build_compressed_abstracts` + `_build_evidence_pools`） | **✅ 完成** | `src/research/services/compression.py` |
| Phase 3 | `extract_compression_node` 节点 | **✅ 完成** | `src/research/graph/nodes/extract_compression.py` |
| Phase 4 | `draft_node` 使用压缩上下文 | **✅ 完成** | `src/research/graph/nodes/draft.py` |
| Phase 5 | Research Graph 注册新节点 | **✅ 完成** | `src/research/graph/builder.py` |
| Phase 6 | AgentState 增加字段 | **✅ 完成** | `src/graph/state.py` |

---

## 一、现状分析：当前没有任何上下文压缩

### 1.1 上下文流经报告生成管道的数据量

```
用户查询
    ↓
clarify_node          → 短文本（ResearchBrief），无需压缩
    ↓
search_plan_node      → SearchPlan（短），无需压缩
    ↓
search_node           → 返回 RagResult（含 paper_candidates，原始 hits）
    ↓
extract_node          → PaperCards（最多 30 张）  ←─┐
    ↓                                        │
draft_node  ────────────────────────────────┘
    │
    └→ 传入 _build_draft_report:
        cards[:20] × 每张完整 abstract (~1500 chars)
        + system prompt (~2000 chars)
        + brief_ctx (~500 chars)
        + output (~8000 chars)
        ─────────────────────────────────
        总计 ~26k tokens → 直接送入 LLM
        ⚠️ 无任何压缩
```

### 1.2 当前唯一的"截断"机制（不是压缩）

| 位置 | 截断策略 | 效果 |
|------|---------|------|
| `extract_node` | `MAX_EXTRACT_CANDIDATES = 30` | 限制候选数量，但每张卡片内容不变 |
| `extract_node` | 每批 3 张并行 LLM 抽取 | 抽取本身有截断（`abstract[:1500]`），但仅限 DeepXiv brief |
| `draft_node` | `cards[:20]` | 丢弃第 20 张之后的卡片 |
| `AnalystAgent` | `cards[:10]` | 丢弃第 10 张之后的卡片 |
| `writing_scaffold_generator` | `paper_cards[:10]` | 只传 10 篇论文 |

这些都是**硬截断**，不是**压缩**：
- 丢弃的卡片内容永久丢失
- 保留下来的卡片仍然携带完整 abstract
- LLM 必须处理所有原始内容才能理解

### 1.3 论文摘要的 token 分布

```
每张 PaperCard 的 abstract 长度分布（估算）：
- 短 abstract：~500 chars ≈ 125 tokens
- 正常 abstract：~1500 chars ≈ 375 tokens  ← 典型值
- 长 abstract：~3000+ chars ≈ 750 tokens

20 张卡片的 token 分布：
- 最佳情况（20 × 500）：~10k tokens
- 典型情况（20 × 1500）：~30k tokens  ⚠️ 超限
- 最坏情况（混合长尾）：~50k tokens  ⚠️ 严重超限
```

---

## 二、可用但未集成的压缩原语

系统中存在多个潜在的压缩组件，但都没有被集成到报告生成流程中：

### 2.1 `summarize_hits` — 最接近压缩的现有工具

**位置**: `src/tools/search_tools.py` 第 312-336 行

```312:336:src/tools/search_tools.py
@tool
def summarize_hits(results: str) -> str:
    """
    对一组搜索结果进行摘要归纳，返回：
    覆盖主题、高质量结果、低质量结果、缺失角度。
    输入为原始搜索结果文本，输出为结构化摘要。
    """
    llm = build_quick_llm(...)
    prompt = (
        f"以下是搜索结果：\n{results}\n\n"
        "请对上述搜索结果进行摘要分析，输出以下内容（严格 JSON）：\n"
        "{\n"
        '  "covered_topics": [...],\n'
        '  "high_quality_hits": [...],\n'
        '  "low_quality_hits": [...],\n'
        '  "missing_angles": [...]\n'
        "}"
    )
```

**用途**: 对原始搜索结果进行摘要，但目前仅被 `SearchPlanAgent` 在多轮迭代中使用（第 115、285 行）。

**未集成到**: `draft_node`、`extract_node`、`AnalystAgent`。

### 2.2 `FineChunker` — 细粒度分块（用于文档，不用于报告）

**位置**: `src/corpus/ingest/fine_chunker.py`

```
FineChunker 的策略：
- 段落切分（paragraph-level）
- Sentence sliding window（2-5 句/组）
- Overlap（前后各 60 chars）
- 目标：350 chars / chunk
```

**用途**: 文档 INGESTION 时对 PDF 进行细粒度分块，服务于 evidence retrieval。

**问题**: 仅用于文档入库后的检索，不用于报告生成阶段。

### 2.3 `comparison_matrix_builder` — 潜在的结构化压缩

**位置**: `src/skills/research_skills.py` 第 238-327 行

**作用**: 将 20 张卡片转换为结构化对比矩阵：
```python
{"rows": [
  {"paper": "Paper Title", "methods": "...", "datasets": "...", "benchmarks": "...", "limitations": "..."},
  ...
]}
```

**压缩效果**: 
- 原始：20 × ~1500 chars = ~30k chars
- 压缩后：结构化字段，每篇 ~200 chars = ~4k chars
- **压缩率约 87%**

**问题**: 
- `AnalystAgent` 中只处理前 10 篇（`cards[:10]`）
- `writing_scaffold_generator` 也只处理前 10 篇（`paper_cards[:10]`）
- 这两个 skills 根本没有被 `draft_node` 调用

### 2.4 `writing_scaffold_generator` — 潜在的大纲压缩

**位置**: `src/skills/research_skills.py` 第 418-510 行

**作用**: 生成结构化写作框架：
```python
{"sections": {"introduction": ["paragraph outline 1", ...], ...}, "outline": [...], "writing_guidance": "..."}
```

**压缩效果**: 将原始论文内容压缩为写作指引，但只生成大纲，不包含论文具体内容。

**问题**: 只生成大纲，draft 阶段仍然需要完整论文内容。

---

## 三、压缩架构设计

### 3.1 多层压缩管道

```
┌─────────────────────────────────────────────────────────────────────┐
│                    上下文压缩管道（Compression Pipeline）                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 0: Raw Input                                                │
│  20 × paper_cards (每张 ~1500 chars abstract) = ~30k chars        │
│                                                                     │
│         ↓ compress_paper_cards(paper_cards[:20])                    │
│                                                                     │
│  Layer 1: Structured Summary ──────────────────────────────────────  │
│  - extract_structured_cards(): 从 abstract 提取结构化字段            │
│  - build_comparison_matrix(): 对比矩阵（87% 压缩率）               │
│  - build_taxonomy(): 论文分类（识别技术路线/子领域）               │
│  Output: ~4k chars + 结构化矩阵                                    │
│                                                                     │
│         ↓ build_evidence_pool(论文分组 + 矩阵)                      │
│                                                                     │
│  Layer 2: Section-level Evidence Pool ─────────────────────────────  │
│  - 对每个 section 预分配 token 预算                                 │
│  - 中心论文（被多篇引用）分配更多 token                             │
│  - 边缘论文（孤立）截断至核心声明                                   │
│  - 跨论文共同信息合并（避免重复）                                   │
│  Output: 每个 section 的 evidence pool（动态大小）                 │
│                                                                     │
│         ↓ write_with_evidence(section_pool)                         │
│                                                                     │
│  Layer 3: Per-Section Drafting ───────────────────────────────────  │
│  - 为每个 section 独立分配 token 预算（见下表）                      │
│  - 按 pool 分配比例分配 token                                      │
│  - 无需超量读取，一次 LLM 调用完成单 section 写作                   │
│  Output: 各 section 草稿                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Token 预算分配

| Section | 当前约束 | 建议约束 | 说明 |
|---------|---------|---------|------|
| introduction | 800-1200 chars | 2500-3500 chars | 发展脉络+动机+贡献+roadmap |
| background | 600-1000 chars | 1500-2000 chars | 基础概念+问题定义 |
| taxonomy | 1000-1500 chars | 2000-3000 chars | 分类体系+代表性工作 |
| methods | 1200-1800 chars | 3000-4000 chars | 各方法详细对比 |
| datasets | 600-1000 chars | 1000-1500 chars | 基准数据集表格 |
| evaluation | 800-1200 chars | 1500-2000 chars | 性能对比+数值 |
| discussion | 600-900 chars | 1500-2000 chars | 交叉主题+trade-off |
| future_work | 500-800 chars | 800-1200 chars | 开放问题+方向 |
| conclusion | 300-500 chars | 500-800 chars | 总结 |

### 3.3 压缩算法设计

#### 算法 A：`build_taxonomy` — 论文分类压缩

**目的**: 将 20+ 篇论文按技术路线/子领域分类，减少"平铺罗列"

**输入**: `List[PaperCard]`
**输出**: `Taxonomy` 结构

```python
@dataclass
class Taxonomy:
    categories: list[TaxonomyCategory]
    cross_category_themes: list[str]   # 跨分类主题（最重要）
    timeline: list[str]                 # 技术发展时间线
    key_papers: list[str]              # 必引用的关键论文

@dataclass
class TaxonomyCategory:
    name: str                           # "Benchmark驱动" / "Agent架构" / ...
    description: str
    papers: list[str]                   # 该分类下的论文标题
    key_characteristics: list[str]       # 该分类的核心特征
    shared_insights: list[str]           # 该分类内跨论文的共同发现
    conflicts: list[str]                # 分类内论文间的冲突结论
```

**LLM 调用**:
```
SYSTEM: "你是一个论文分类专家..."
INPUT: 20 张卡片
OUTPUT: Taxonomy JSON
压缩率: ~90%（从 30k chars → ~3k chars）
```

#### 算法 B：`build_compressed_abstracts` — 论文摘要压缩

**目的**: 将每张卡片压缩到 ~300 chars（保留核心发现）

**输入**: `List[PaperCard]`
**输出**: `List[CompressedCard]`

```python
@dataclass
class CompressedCard:
    title: str
    arxiv_id: str
    core_claim: str         # 核心发现（一句话）
    method_type: str         # 方法类型
    key_result: str          # 关键数值结果（如果有）
    role_in_taxonomy: str    # 在分类中的角色
    connections: list[str]   # 与其他论文的关系/对比
```

**压缩率**: ~80%（从 ~1500 chars → ~300 chars / 张）

#### 算法 C：`build_evidence_pool` — Section 级 Evidence 池

**目的**: 按 section 分配 evidence，避免一篇论文被所有 section 全文引用

**输入**: `Taxonomy` + `List[CompressedCard]`
**输出**: `Dict[SectionName, EvidencePool]`

```python
@dataclass
class EvidencePool:
    section: str
    token_budget: int           # 该 section 的 token 预算
    papers: list[PoolEntry]    # 分配给该 section 的论文

@dataclass
class PoolEntry:
    card: CompressedCard
    allocated_chars: int        # 分配给该论文的字符数
    focus_aspect: str           # 该 section 关注该论文的哪个方面
```

**分配策略**:
1. 识别每篇论文在不同 section 的相关性（通过 taxonomy 匹配）
2. 中心论文（被多个分类引用）→ 所有相关 section 都分配 evidence
3. 边缘论文（仅属于一个分类）→ 只在相关 section 出现
4. token 预算按 section 重要性 + 相关论文数量动态分配

---

## 四、实施路径

### 阶段 1：在 `extract_node` 后、draft_node 前插入压缩层（P0）

```
extract_node  →  extract_compression_node  →  draft_node
```

新增 `extract_compression_node`:

```python
def extract_compression_node(state: dict) -> dict:
    """在 extract 和 draft 之间插入上下文压缩。"""
    paper_cards = state.get("paper_cards", [])
    brief = state.get("brief")
    
    # Step 1: 构建 Taxonomy
    taxonomy = _build_taxonomy(paper_cards, brief)
    
    # Step 2: 压缩论文摘要
    compressed = _build_compressed_abstracts(paper_cards, taxonomy)
    
    # Step 3: 构建 Per-Section Evidence Pool
    pools = _build_evidence_pools(compressed, taxonomy, brief)
    
    return {
        "taxonomy": taxonomy,
        "compressed_cards": compressed,
        "evidence_pools": pools,
    }
```

**单次 LLM 调用**：只需 1 次 LLM 调用完成 taxonomy + compressed abstracts（可以合并）。

### 阶段 2：修改 `draft_node` 使用压缩后的上下文（P0）

将 `_build_draft_report` 从"传入原始 20 张卡片"改为"传入 evidence_pools + compressed_cards"。

```python
def _build_draft_report(cards: list[Any], brief: Any | None, 
                       taxonomy: Taxonomy | None = None,
                       compressed_cards: list[CompressedCard] | None = None,
                       evidence_pools: dict[str, EvidencePool] | None = None) -> DraftReport:
    """使用压缩后的上下文生成报告。"""
    
    # 为每个 section 独立生成，使用对应的 evidence pool
    sections = {}
    for section_name, pool in (evidence_pools or {}).items():
        section_text = _write_section(
            section_name,
            pool=pool,
            taxonomy=taxonomy,
            brief=brief,
        )
        sections[section_name] = section_text
    
    # Claims 和 Citations 从 taxonomy + compressed_cards 生成
    ...
```

### 阶段 3：将 `comparison_matrix_builder` 深度集成（P1）

在 `taxonomy` 构建后，调用 `comparison_matrix_builder` 生成对比矩阵，作为 taxonomy 的补充证据：

```python
def _build_taxonomy_with_matrix(cards, brief):
    taxonomy = _build_taxonomy(cards, brief)          # 结构化分类
    matrix = comparison_matrix_builder({                # 对比矩阵
        "paper_cards": cards[:20],
        "compare_dimensions": ["methods", "datasets", "benchmarks"],
        "format": "json"
    }, {})
    return taxonomy, matrix
```

### 阶段 4：修改 `AnalystAgent` RVA 流水线（P1）

修复 `AnalystAgent` 的现有问题：
1. 移除 `cards[:10]` 硬截断，改为处理全部 20 张
2. `comparison_matrix_builder` 失败时回退到硬编码矩阵而非空 dict
3. 将 `build_graph()` 从硬编码 5 步链改为置信度驱动的条件边

### 阶段 5：集成 `summarize_hits` 到搜索结果聚合（P2）

在 `search_node` 的 `_ingest_paper_candidates` 阶段，对检索到的论文列表调用 `summarize_hits` 生成元摘要，减少 extract_node 的原始输入量。

---

## 五、Token 预算管理机制

### 5.1 全局 Budget Tracker

```python
class ContextBudget:
    """
    管理整个报告生成的 token 预算。
    
    DeepSeek context window: ~128k tokens
    目标使用率: 70-80%（保留 buffer 给 system prompt + output）
    可用 tokens: ~90k tokens
    
    分配：
    - system prompt: ~3k tokens
    - brief context: ~0.5k tokens
    - compressed_cards: ~8k tokens (20 × ~400)
    - taxonomy: ~3k tokens
    - comparison_matrix: ~5k tokens
    - per-section evidence: ~60k tokens (各 section 按需分配)
    - output buffer: ~10k tokens
    """
    
    TOTAL_BUDGET = 90000  # tokens
    SECTION_BUDGETS = {
        "introduction": 15000,
        "background": 10000,
        "taxonomy": 15000,
        "methods": 20000,
        "datasets": 8000,
        "evaluation": 10000,
        "discussion": 8000,
        "future_work": 6000,
        "conclusion": 4000,
    }
    
    def allocate(self, section: str, evidence: list[CompressedCard]) -> str:
        """将压缩后的 evidence 分配给 section，超量时截断。"""
        budget = self.SECTION_BUDGETS.get(section, 5000)
        allocated = self._pack_evidence(evidence, budget)
        return allocated  # 返回压缩后的文本
```

### 5.2 动态调整策略

```
当 paper_cards > 20 时：
- 优先保留：被 taxonomy 标记为"关键论文"的卡片
- 次优先：属于多个 taxonomy 类别的卡片（中心论文）
- 最后截断：孤立论文（只属于一个类别且与其他论文无明显关联）

当单个 abstract 过长时：
- 截断策略：保留前 500 chars（方法描述）+ 后 200 chars（结果/结论）
- 过滤策略：移除 Introduction 中的背景描述（与报告 introduction 重复）
```

---

## 六、与其他模块的关系

```
压缩管道与其他模块的关系：

┌─────────────────────────────────────────────────────┐
│                                                     │
│  ClarifyAgent ──→ ResearchBrief ──→ Taxnomy 构建   │
│                                             ↓       │
│  SearchNode ──→ RagResult ──→ ExtractNode ──→ PaperCards
│                                             ↓       │
│                              ExtractCompressionNode   │
│                              (新增： Taxonomy +      │
│                               CompressedCards +      │
│                               EvidencePools)         │
│                                             ↓       │
│  SkillOrchestrator ──→ SkillInvoker ──→ comparison_matrix_builder
│                                             ↓       │
│                                       DraftNode      │
│                                             ↓       │
│                                     ReviewNode      │
│                                             ↓       │
│                                   GroundService     │
│                              (resolve_citations +  │
│                               verify_claims)        │
└─────────────────────────────────────────────────────┘

关键依赖：
- 需要 ClarifyAgent 的 brief 上下文（用于 Taxonomy 构建）
- 需要 ExtractNode 的 structured cards（用于压缩）
- 需要 SkillOrchestrator 集成 comparison_matrix_builder
- 为 ReviewNode 提供更结构化的 evidence（改善 citation 覆盖率）
```

---

## 七、实现优先级与验收标准

### P0：Extract Compression Node + Draft Node 重构

**目标**: 在一次 LLM 调用中完成 taxonomy 构建 + 论文摘要压缩

**验收标准**:
- [ ] `extract_compression_node` 成功生成 Taxonomy + CompressedCards
- [ ] `_build_draft_report` 使用压缩后上下文，报告不再出现"摘要复读"
- [ ] 每个 section 独立生成，不在一段 LLM 调用中生成全部
- [ ] Introduction 字数达到 2000+ chars
- [ ] 各 section 引用的论文不重复（通过 evidence pool 隔离）

### P1：Comparison Matrix 集成 + AnalystAgent 修复

**验收标准**:
- [ ] `comparison_matrix_builder` 在压缩管道中被调用
- [ ] `AnalystAgent` 处理全部 20 张卡片（非 `[:10]`）
- [ ] `AnalystAgent` 的 `build_graph()` 支持置信度回退

### P2：动态 Token Budget 管理

**验收标准**:
- [ ] `ContextBudget` 类实现，支持动态分配
- [ ] 当论文数 > 20 时，自动触发压缩而非硬截断
- [ ] 关键论文在所有相关 section 中都被引用

---

## 八、相关文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/research/graph/nodes/extract_compression.py` | 新建 | 压缩节点 |
| `src/research/graph/nodes/draft.py` | 修改 | 使用压缩上下文 |
| `src/research/graph/builder.py` | 修改 | 注册新节点 |
| `src/research/agents/analyst_agent.py` | 修改 | 移除 `[:10]` 硬截断 |
| `src/research/services/compression.py` | 新建 | 压缩算法核心实现 |
| `src/models/research.py` | 新建 | `Taxonomy`, `CompressedCard`, `EvidencePool` 模型 |
| `src/skills/research_skills.py` | 修改 | 集成到压缩管道 |
| `tests/research/graph/nodes/test_extract_compression.py` | 新建 | 压缩节点测试 |
