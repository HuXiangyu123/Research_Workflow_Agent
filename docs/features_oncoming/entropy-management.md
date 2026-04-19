# Entropy Management — Agent 系统熵管理

> 生成时间：2026-04-14
> 状态：**Phase 1 已实现**（见下文）
> 优先级：P1（Harness Engineering 缺失的最后一块核心拼图）

---

## 实现状态

| 阶段 | 内容 | 状态 | 文件 |
|------|------|------|------|
| Phase 1 | `EntropyReport` + `DriftReport` 数据模型 | **✅ 完成** | `src/entropy/scanner.py` |
| Phase 2 | `ConstraintViolationDetector`（检测 SQLite 引入） | **✅ 完成** | `src/entropy/detectors/constraint.py` |
| Phase 3 | `DeadCodeDetector`（检测幽灵节点引用） | **✅ 完成** | `src/entropy/detectors/constraint.py` |
| Phase 4 | `DocDriftDetector`（检测文档漂移） | **✅ 完成** | `src/entropy/detectors/constraint.py` |
| Phase 5 | Entropy CLI（`scan` + `check` 命令） | **✅ 完成** | `src/entropy/cli.py` |

---

## 一、背景：为什么 Agent 系统会腐化

### 1.1 Entropy（熵）的定义

在 Harness Engineering 框架中，**Entropy** 是指 AI Agent 系统运行一段时间后，代码库和文档逐渐偏离原始设计意图的累积效应。与热力学的熵增类似：没有主动做功（清理），系统自发趋向混乱。

### 1.2 Agent 系统熵的来源

```
┌─────────────────────────────────────────────────────────┐
│                    Entropy 的四大来源                    │
├─────────────────┬───────────────────────────────────────┤
│  文档漂移        │  代码改了，文档没改；文档改了，代码没改     │
├─────────────────┼───────────────────────────────────────┤
│  模式不一致      │  Agent A 生成的代码风格 ≠ Agent B 的     │
├─────────────────┼───────────────────────────────────────┤
│  死代码积累      │  重命名节点后旧代码路径仍在，Agent 仍会撞   │
├─────────────────┼───────────────────────────────────────┤
│  约束侵蚀        │  新增 import 绕过 .cursorignore 规则     │
└─────────────────┴───────────────────────────────────────┘
```

### 1.3 当前项目的熵证据

| 熵类型 | 证据 | 影响 |
|--------|------|------|
| 文档漂移 | `docs/design_version/` 下有多个版本（2026-02-15、2026-03-29），结构不一致 | Agent 无法判断哪个是真实架构 |
| 文档漂移 | `docs/active/` 和 `docs/design_version/` 混用，Phase 2/3/4 文档散落各处 | 新 Agent 困惑 |
| 模式不一致 | `analyst_agent.py` 有 `build_graph()`，但 `retriever_agent.py` 只有 import | 同为 V2 Agent，实现形态不一致 |
| 死代码 | `src/research/graph/nodes/` 下部分节点未被 `LEGACY_NODE_TARGETS` 引用但文件存在 | Agent 可能误调 |
| 约束侵蚀 | `.cursorignore` 被修改（git status 显示 `M .cursorignore`）| 规则被绕过 |

---

## 二、设计目标

Entropy Management 系统需要实现三个核心能力：

1. **检测（Detect）**：主动发现文档漂移、代码不一致、约束违反
2. **清理（Clean）**：自动或半自动修复熵增
3. **预防（Prevent）**：在 CI 层面阻止新的熵增

---

## 三、系统架构

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Entropy Management System                      │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ EntropyScanner│→  │ EntropyReport │→  │ Scheduled Cleanup Agents │  │
│  │  (检测层)    │    │  (报告层)     │    │  (清理层)                │  │
│  └──────┬──────┘    └──────┬──────┘    └───────────┬─────────────┘  │
│         │                  │                       │                │
│         ▼                  ▼                       ▼                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               Hooks / CI Integration (预防层)                 │  │
│  │         Pre-commit + CI Pipeline + Agent System Hooks         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│                 ┌─────────────────┐                                 │
│                 │ Entropy Dashboard│ (前端展示)                       │
│                 └─────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
src/entropy/
├── __init__.py
├── scanner.py          # 检测器核心
├── detectors/
│   ├── __init__.py
│   ├── doc_drift.py     # 文档漂移检测
│   ├── style_drift.py   # 代码风格不一致检测
│   ├── dead_code.py     # 死代码检测
│   ├── constraint.py    # 约束违反检测
│   └── artifact.py       # Agent 产物质量检测
├── cleaners/
│   ├── __init__.py
│   ├── doc_cleaner.py   # 文档修复
│   ├── style_cleaner.py # 代码格式化
│   └── dead_code_cleaner.py
├── scheduler.py         # 调度器
├── report.py            # 报告生成
└── hooks.py             # CI / pre-commit 集成

src/api/routes/
└── entropy.py           # Entropy Management API

tests/entropy/
└── test_*.py

scripts/
└── entropy_scan.py      # CLI 入口
```

---

## 四、检测器设计（Detect）

### 4.1 文档漂移检测（Doc Drift Detector）

**问题**：代码改了，文档没改；或文档改了，代码没改。

**检测策略**：

```python
class DocDriftDetector:
    """检测代码与文档之间的不一致。"""

    def __init__(self):
        self.rules = [
            # 规则 1：src/ 下的模块必须在 docs/ 有对应说明
            DriftRule(
                pattern="src/research/agents/*.py",
                expected_doc="docs/active/phase/{agent_name}.md",
                drift_type="missing_doc",
            ),
            # 规则 2：.cursorrules 中声明的约束必须有 linter 支持
            DriftRule(
                pattern=".cursorrules",
                expected_enforcement="Always use PostgreSQL, never SQLite",
                check_enforced=True,
                drift_type="unenforced_constraint",
            ),
            # 规则 3：CANONICAL_NODE_ORDER 中列出的节点必须存在
            DriftRule(
                pattern="src/research/agents/supervisor.py",
                check_list="CANONICAL_NODE_ORDER",
                check_exists="src/research/graph/nodes/{node}.py",
                drift_type="missing_node_file",
            ),
        ]

    def scan(self) -> list[DriftReport]:
        """扫描所有漂移问题。"""
        ...

@dataclass
class DriftReport:
    drift_type: str          # missing_doc | unenforced_constraint | missing_node_file
    source_file: str         # 违规的文件
    expected_state: str      # 期望状态
    actual_state: str        # 实际状态
    severity: Literal["critical", "warning", "info"]
    fix_suggestion: str      # 修复建议
```

**当前项目中的检测目标**：

| 漂移类型 | 具体案例 | 期望状态 |
|----------|----------|----------|
| 缺失文档 | `src/research/agents/analyst_agent.py` 没有对应 `docs/active/phase/` 文档 | 每个 V2 Agent 有对应说明 |
| 约束违反 | `.cursorrules` 说 PostgreSQL only，但代码中有其他路径 | linter 能检测 |
| 节点缺失 | `CANONICAL_NODE_ORDER` 列了 `persist_artifacts`，但文件可能不存在 | 文件存在且可导入 |

### 4.2 代码风格不一致检测（Style Drift Detector）

**问题**：不同 Agent 生成的代码风格不一致。

**检测策略**：

```python
class StyleDriftDetector:
    """检测 Agent 产物中的代码风格漂移。"""

    def __init__(self):
        # 定义期望风格基准（从现有好代码中提取）
        self.baseline = StyleBaseline(
            import_order=["__future__", "stdlib", "third_party", "local"],
            max_line_length=120,
            docstring_style="google",
            type_hint_coverage=0.8,   # 80% 参数有类型提示
            no_fromtyping_import="TypedDict",  # 应该用 typing.TypedDict
        )

    def check_agent_artifacts(self, agent_dir: Path) -> StyleDriftReport:
        """检查 Agent 产物目录的风格一致性。"""
        ...

@dataclass
class StyleDriftReport:
    agent: str
    metrics: dict[str, float]  # actual vs baseline
    violations: list[StyleViolation]
    score: float  # 0-1，一致性得分
```

**检测指标**：

| 指标 | 含义 | 检测方法 |
|------|------|----------|
| `type_hint_coverage` | 参数/返回值类型提示覆盖率 | AST 解析 |
| `import_order_score` | import 顺序规范程度 | 正则匹配 |
| `docstring_score` | 文档字符串覆盖率 | AST 解析 |
| `consistency_with_baseline` | 与基准风格的相似度 | tree-sitter 对比 |

### 4.3 死代码检测（Dead Code Detector）

**问题**：代码改了，但旧路径仍存在，Agent 可能误调。

**检测策略**：

```python
class DeadCodeDetector:
    """检测无法到达的代码路径。"""

    def scan_unreachable_nodes(self) -> list[DeadCodeReport]:
        """检测 LEGACY_NODE_TARGETS / V2_AGENT_TARGETS 中引用的节点是否都存在。"""
        reports = []

        # 检查 supervisor 中的节点引用
        supervisor = Path("src/research/agents/supervisor.py").read_text()

        # 提取 LEGACY_NODE_TARGETS
        legacy_nodes = self._extract_dict_values(supervisor, "LEGACY_NODE_TARGETS")
        for node in legacy_nodes:
            module_path = f"src/research/graph/nodes/{node}.py"
            if not Path(module_path).exists():
                reports.append(DeadCodeReport(
                    node_name=node,
                    referenced_by="LEGACY_NODE_TARGETS",
                    actual_path=None,
                    severity="critical",
                    suggestion=f"移除 LEGACY_NODE_TARGETS 中对 {node} 的引用",
                ))

        # 检查 V2_AGENT_TARGETS
        v2_nodes = self._extract_dict_values(supervisor, "V2_AGENT_TARGETS")
        for node in v2_nodes:
            module_path = f"src/research/agents/{node.replace('_', '')}_agent.py"
            if not Path(module_path).exists():
                reports.append(DeadCodeReport(...))

        return reports

    def scan_orphaned_files(self) -> list[DeadCodeReport]:
        """检测没有被任何地方引用的文件。"""
        all_imports = self._extract_all_imports()
        orphaned = []

        for py_file in Path("src").rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            imports_this = self._extract_imports(py_file)
            if not any(imp in all_imports or self._file_name_in_imports(py_file, all_imports)
                       for imp in imports_this):
                # 检查是否是 CLI 入口或测试直接引用
                if not self._is_entry_point(py_file):
                    orphaned.append(...)
        return orphaned
```

### 4.4 约束违反检测（Constraint Violation Detector）

**问题**：代码违反了 `.cursorrules` 或 `AGENTS.md` 中声明的约束。

```python
class ConstraintViolationDetector:
    """检测对项目硬约束的违反。"""

    HARD_CONSTRAINTS = [
        # 约束 1：不能有 SQLite
        Constraint(
            id="no_sqlite",
            description="禁止引入 SQLite 数据库或 sqlite:/// URL",
            check=lambda f: "sqlite:///" not in f.read_text()
                          and ".sqlite" not in f.name,
            severity="critical",
        ),
        # 约束 2：所有持久化必须走 PostgreSQL
        Constraint(
            id="postgres_persistence",
            description="所有长期持久化必须使用 DATABASE_URL",
            check=self._check_persistence_layer,
            severity="critical",
        ),
        # 约束 3：.env 必须在脚本中显式加载
        Constraint(
            id="explicit_dotenv",
            description="脚本和测试必须显式 load_dotenv('.env')",
            check=self._check_dotenv_loading,
            severity="warning",
        ),
        # 约束 4：V2 Agent 必须实现 build_graph()
        Constraint(
            id="agent_has_graph",
            description="V2_AGENT_TARGETS 中的 Agent 必须有 build_graph() 方法",
            check=self._check_agent_graph,
            severity="warning",
        ),
    ]
```

### 4.5 Agent 产物质量检测（Artifact Quality Detector）

**问题**：Agent 生成的产物（artifacts）质量不一致或退化。

```python
class ArtifactQualityDetector:
    """检测 Agent 产物（artifacts）质量问题。"""

    def check_report_quality(self, report_path: Path) -> QualityReport:
        """检测报告质量是否达标。"""
        content = report_path.read_text()
        return QualityReport(
            has_abstract=len(content.split("## Abstract")) > 1,
            has_citations=self._has_citations(content),
            citation_count=self._count_citations(content),
            section_count=len([h for h in content.split("\n") if h.startswith("## ")]),
            avg_section_length=self._avg_section_length(content),
            quality_score=self._compute_quality_score(...),
        )

    def check_artifact_bloat(self) -> list[BloatReport]:
        """检测 LLM 调用结果是否过大（token 浪费）。"""
        # 检查 paper_cards 是否超过 20 条（项目限制）
        # 检查 comparison_matrix 是否过大
        # 检查是否有重复的 artifacts
        ...
```

---

## 五、清理器设计（Clean）

### 5.1 清理器类型

```
┌──────────────────────────────────────────────────────────────┐
│                    Cleaners（清理层）                        │
├──────────────────┬──────────────────┬───────────────────────┤
│   DocCleaner     │  StyleCleaner    │  DeadCodeCleaner     │
├──────────────────┼──────────────────┼───────────────────────┤
│ • 补全缺失文档    │ • 统一 import 顺序│ • 删除孤立文件        │
│ • 删除过时文档    │ • 格式化代码      │ • 清理未引用节点      │
│ • 同步文档版本    │ • 补全类型提示    │ • 移除 orphan imports │
│ • 合并重复文档    │ • 补全 docstring  │                      │
└──────────────────┴──────────────────┴───────────────────────┘
```

### 5.2 DocCleaner

```python
class DocCleaner:
    """文档清理：补全、删除过时文件、同步版本。"""

    def generate_missing_docs(self, drift_reports: list[DriftReport]) -> list[FileChange]:
        """为缺失的文档生成占位符。"""
        changes = []
        for report in drift_reports:
            if report.drift_type == "missing_doc":
                template = self._get_doc_template(report.source_file)
                changes.append(FileChange(
                    path=report.expected_doc,
                    action="create",
                    content=template,
                    reason=f"文档缺失：{report.source_file}",
                ))
        return changes

    def prune_obsolete_docs(self) -> list[FileChange]:
        """删除过时文档（如设计版本归档过期的旧版本）。"""
        # 删除 6 个月前的 design_version 文档
        cutoff = datetime.now() - timedelta(days=180)
        changes = []
        for doc in Path("docs/design_version/").rglob("*.md"):
            mtime = datetime.fromtimestamp(doc.stat().st_mtime)
            if mtime < cutoff:
                changes.append(FileChange(
                    path=str(doc),
                    action="delete",
                    content=None,
                    reason=f"文档过期（{mtime.date()}）",
                ))
        return changes

    def sync_doc_structure(self) -> list[FileChange]:
        """统一 docs/ 目录结构，确保 Agent 可预测文档位置。"""
        # 确保 docs/active/phase/ 下每个 Phase 有对应文档
        # 确保每个 V2 Agent 有对应 docs/research/agents/ 文档
        ...
```

### 5.3 DeadCodeCleaner

```python
class DeadCodeCleaner:
    """死代码清理：删除孤立文件、清理无效引用。"""

    def remove_orphaned_files(self, reports: list[DeadCodeReport]) -> list[FileChange]:
        """删除孤立文件。"""
        changes = []
        for report in reports:
            if report.drift_type == "orphaned_file":
                changes.append(FileChange(
                    path=report.source_file,
                    action="delete",
                    reason=f"孤立文件（未被任何地方引用）：{report.source_file}",
                ))
        return changes

    def fix_missing_node_references(self, reports: list[DeadCodeReport]) -> list[FileChange]:
        """清理对不存在节点的引用。"""
        changes = []
        supervisor_path = Path("src/research/agents/supervisor.py")

        for report in reports:
            if "missing_node" in report.drift_type:
                changes.append(FileChange(
                    path=str(supervisor_path),
                    action="edit",
                    old_string=report.referenced_by,
                    new_string="# " + report.referenced_by + " (removed by entropy cleaner)",
                    reason=f"引用了不存在的节点：{report.node_name}",
                ))
        return changes
```

---

## 六、调度器设计（Schedule）

### 6.1 调度策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    清理任务调度策略                              │
├─────────────────┬───────────────────────────────────────────────┤
│  类型           │  触发条件                                      │
├─────────────────┼───────────────────────────────────────────────┤
│  On-commit      │  每次 git commit 后自动扫描，检测新增熵         │
├─────────────────┼───────────────────────────────────────────────┤
│  Daily          │  每天凌晨 2:00 运行全量扫描 + 报告               │
├─────────────────┼───────────────────────────────────────────────┤
│  On-PR          │  PR 打开时运行 PR 范围内的 Entropy 扫描         │
├─────────────────┼───────────────────────────────────────────────┤
│  Manual         │  开发者手动触发（API 或 CLI）                    │
├─────────────────┼───────────────────────────────────────────────┤
│  On-demand      │  Agent 发现问题时主动触发（通过 hooks）          │
└─────────────────┴───────────────────────────────────────────────┘
```

### 6.2 调度器实现

```python
class EntropyScheduler:
    """熵管理调度器。"""

    def __init__(self, scanner: EntropyScanner, cleaners: list[EntropyCleaner]):
        self.scanner = scanner
        self.cleaners = cleaners
        self._schedule: dict[str, callable] = {
            "on_commit": self._on_commit,
            "daily": self._daily_scan,
            "on_pr": self._on_pr_scan,
            "manual": self._manual_scan,
        }

    async def run_on_commit(self, changed_files: list[str]) -> EntropyReport:
        """git commit 钩子触发：只扫描变更文件。"""
        reports = self.scanner.scan_files(changed_files)
        return self._build_report(reports, trigger="on_commit")

    async def run_daily(self) -> EntropyReport:
        """每日全量扫描。"""
        reports = self.scanner.scan_all()
        changes = self._apply_auto_fixes(reports)
        return self._build_report(reports, changes, trigger="daily")

    async def run_on_pr(self, pr_diff: str, base_branch: str) -> EntropyReport:
        """PR 触发：扫描 PR 范围内的新增熵。"""
        changed_files = self._extract_changed_files(pr_diff)
        # 特别关注：PR 引入了新的约束违反
        new_violations = self.scanner.scan_files(changed_files)
        return self._build_report(new_violations, trigger="on_pr")

    def _apply_auto_fixes(self, reports: list[DriftReport]) -> list[FileChange]:
        """
        自动应用可安全修复的变更。
        规则：只有 severity=info 或 confirmed_safe=True 的才自动修复。
        """
        changes = []
        for report in reports:
            if report.severity == "info" and report.auto_fixable:
                for cleaner in self.cleaners:
                    if cleaner.can_handle(report):
                        changes.extend(cleaner.fix(report))
        return changes
```

---

## 七、报告设计（Report）

### 7.1 Entropy 报告结构

```python
@dataclass
class EntropyReport:
    timestamp: datetime
    trigger: str  # on_commit | daily | on_pr | manual
    summary: EntropySummary
    drift_reports: list[DriftReport]
    style_reports: list[StyleDriftReport]
    dead_code_reports: list[DeadCodeReport]
    constraint_reports: list[ConstraintViolationReport]
    quality_reports: list[QualityReport]
    auto_fix_changes: list[FileChange]
    pending_changes: list[FileChange]  # 需要人工审查的
    entropy_score: float  # 0-100，越低越好

@dataclass
class EntropySummary:
    total_issues: int
    critical: int
    warning: int
    info: int
    entropy_delta: float  # 与上次扫描相比的变化
```

### 7.2 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/entropy/scan` | POST | 手动触发扫描 |
| `/api/v1/entropy/report` | GET | 获取最新报告 |
| `/api/v1/entropy/report/{id}` | GET | 获取指定报告 |
| `/api/v1/entropy/fix` | POST | 应用建议的修复 |
| `/api/v1/entropy/dashboard` | GET | 前端 Dashboard 数据 |

### 7.3 前端 Dashboard

```
┌──────────────────────────────────────────────────────────┐
│  Entropy Dashboard                      [扫描] [设置]     │
├──────────────────────────────────────────────────────────┤
│  当前熵评分：72/100  ⚠️  ↓较上周 -5                       │
├─────────────────────────────┬────────────────────────────┤
│  问题分布                   │  趋势图                     │
│  ● Critical: 3             │  📈 (折线图)                │
│  ● Warning: 12             │                            │
│  ● Info: 45                │                            │
├─────────────────────────────┴────────────────────────────┤
│  最新问题                                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 🔴 Missing: docs/research/agents/analyst_agent.md │ │
│  │    期望: 每个 V2 Agent 有对应文档                   │ │
│  │    建议: 自动生成文档模板 / 手动补全                │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 🟡 Dead code: V2_AGENT_TARGETS 引用不存在的节点    │ │
│  │    search_plan → retriever_agent.py 存在 ✓         │ │
│  │    search → ??? 不存在                             │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 八、预防层设计（Prevent）

### 8.1 Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: entropy-constraint-check
        name: Entropy Constraint Check
        entry: python -m src.entropy.hooks.pre_commit_check
        language: system
        pass_files: true
        stages: [commit]
        types: [python]
```

```python
# src/entropy/hooks.py
def pre_commit_check(files: list[str]) -> int:
    """
    Pre-commit 钩子：检查变更文件是否违反约束。
    返回 0 表示通过，返回 1 表示失败（阻止 commit）。
    """
    scanner = EntropyScanner()
    reports = scanner.scan_files(files)

    critical = [r for r in reports if r.severity == "critical"]
    if critical:
        print(f"❌ Entropy: {len(critical)} critical violations blocked commit:")
        for r in critical:
            print(f"   [{r.drift_type}] {r.source_file}: {r.actual_state}")
        return 1

    warnings = [r for r in reports if r.severity == "warning"]
    if warnings:
        print(f"⚠️  Entropy: {len(warnings)} warnings (commit allowed):")
        for r in warnings[:3]:
            print(f"   [{r.drift_type}] {r.source_file}")

    return 0
```

### 8.2 CI Pipeline 集成

```yaml
# .github/workflows/entropy.yml
name: Entropy Check

on:
  push:
    branches: [main, develop]
  pull_request:
    types: [opened, synchronize]

jobs:
  entropy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Entropy Scanner
        run: python -m src.entropy.cli scan --format json --output entropy-report.json

      - name: Check Critical Violations
        run: |
          CRITICAL=$(jq '[.summary.critical] | add' entropy-report.json)
          if [ "$CRITICAL" -gt 0 ]; then
            echo "❌ Found $CRITICAL critical entropy violations"
            jq '.drift_reports[] | select(.severity == "critical")' entropy-report.json
            exit 1
          fi

      - name: Comment PR with Report
        if: github.event_name == 'pull_request'
        run: python -m src.entropy.cli comment-pr --report entropy-report.json
```

---

## 九、当前项目的具体实施计划

### 9.1 Phase 1：核心框架（P1，最优先）

| 任务 | 内容 | 文件 |
|------|------|------|
| T1.1 | `EntropyScanner` 核心 + `DriftReport` 数据模型 | `src/entropy/scanner.py` |
| T1.2 | `DeadCodeDetector`：检测 `CANONICAL_NODE_ORDER` 引用 vs 文件存在性 | `src/entropy/detectors/dead_code.py` |
| T1.3 | `ConstraintViolationDetector`：检测 SQLite 引入 | `src/entropy/detectors/constraint.py` |
| T1.4 | Entropy CLI：`python -m src.entropy.cli scan` | `src/entropy/__main__.py` |
| T1.5 | `/api/v1/entropy/scan` API | `src/api/routes/entropy.py` |

**立即可修复的当前问题**：

```bash
$ python -m src.entropy.cli scan

[CRITICAL] DeadCodeDetector: V2_AGENT_TARGETS 引用了不存在的节点
  - search_plan → src/research/agents/planner_agent.py ✓ 存在
  - search → src/research/agents/retriever_agent.py ✓ 存在
  - draft → src/research/agents/analyst_agent.py ✓ 存在
  - review → src/research/agents/reviewer_agent.py ✓ 存在

[CRITICAL] ConstraintViolation: retriever_agent.py 只引入了 langgraph 但未实现 build_graph()
  - 建议：补全 retriever_agent.py 的 build_graph() 实现

[WARNING] DocDrift: 缺少文档
  - src/research/agents/analyst_agent.py → docs/research/agents/analyst_agent.md (不存在)
  - src/research/agents/retriever_agent.py → docs/research/agents/retriever_agent.md (不存在)
```

### 9.2 Phase 2：文档管理 + 调度（P2）

| 任务 | 内容 | 文件 |
|------|------|------|
| T2.1 | `DocDriftDetector`：检测代码-文档不一致 | `src/entropy/detectors/doc_drift.py` |
| T2.2 | `DocCleaner`：生成缺失文档、补全模板 | `src/entropy/cleaners/doc_cleaner.py` |
| T2.3 | 每日扫描调度器 | `src/entropy/scheduler.py` |
| T2.4 | Entropy Dashboard API + 前端组件 | `src/api/routes/entropy.py` + `frontend/src/components/EntropyDashboard.tsx` |

### 9.3 Phase 3：风格管理 + 预防（P3）

| 任务 | 内容 | 文件 |
|------|------|------|
| T3.1 | `StyleDriftDetector`：检测 Agent 产物风格不一致 | `src/entropy/detectors/style_drift.py` |
| T3.2 | Pre-commit Hook 集成 | `src/entropy/hooks.py` + `.pre-commit-config.yaml` |
| T3.3 | GitHub Actions CI 集成 | `.github/workflows/entropy.yml` |
| T3.4 | PR Comment 集成（自动在 PR 上报告 Entropy 问题） | `src/entropy/hooks.py` |

### 9.4 Phase 4：产物质量 + 自动化修复（P4）

| 任务 | 内容 | 文件 |
|------|------|------|
| T4.1 | `ArtifactQualityDetector`：检测报告质量退化 | `src/entropy/detectors/artifact.py` |
| T4.2 | 自动修复可安全变更（如 import 排序、docstring 补全） | `src/entropy/cleaners/style_cleaner.py` |
| T4.3 | Entropy Score 趋势追踪 | `src/entropy/report.py` |
| T4.4 | Agent 主动触发清理（通过 Supervisor Hook） | `src/research/agents/supervisor.py` |

---

## 十、与 Harness Engineering 其他组件的关系

```
┌────────────────────────────────────────────────────────────┐
│              Harness Engineering 全景                      │
├──────────────┬────────────────┬───────────────────────────┤
│ Context Eng  │ Arch Constraints│ Entropy Management         │
├──────────────┼────────────────┼───────────────────────────┤
│ .cursorrules │ PostgreSQL-only │ EntropyScanner ← [当前设计] │
│ AGENTS.md    │ CANONICAL_NODE  │ EntropyCleaner             │
│ docs/        │ .cursorignore  │ EntropyScheduler           │
├──────────────┴────────────────┴───────────────────────────┤
│                    Middleware 层（缺失）                   │
│  LocalContextMiddleware ← AgentStartup 时加载 repo 结构     │
│  LoopDetectionMiddleware ← 检测重复编辑（缺失）              │
│  ReasoningSandwichMiddleware ← 不同 token 配比（部分有）    │
│  PreCompletionChecklistMiddleware ← eval/runner.py（有）    │
└────────────────────────────────────────────────────────────┘
```

Entropy Management 填补了 **"防止腐化"** 这个最容易被忽视的环节。

---

## 十一、测试计划

| # | 测试场景 | 验证内容 |
|---|---------|---------|
| T1 | 扫描不存在的节点引用 | 检测到 `V2_AGENT_TARGETS` 中的幽灵引用 |
| T2 | 扫描缺失文档 | 检测到 `retriever_agent.py` 缺少对应文档 |
| T3 | 扫描 SQLite 引入 | 检测到 `sqlite:///` 字符串或 `.sqlite` 文件 |
| T4 | 扫描孤立文件 | 检测到未被引用的 Python 文件 |
| T5 | CLI 全量扫描 | `python -m src.entropy.cli scan` 正常输出报告 |
| T6 | API 扫描触发 | `POST /api/v1/entropy/scan` 返回 EntropyReport |
| T7 | 风格漂移检测 | 对比两个 Agent 生成的代码风格差异 |
| T8 | 自动修复 import 顺序 | 检测 → 自动修复 → 验证顺序正确 |
| T9 | Pre-commit Hook 阻止 | commit 违反约束的文件时被阻止 |
| T10 | Daily 调度器 | 验证每日任务在正确时间触发 |

---

## 十二、相关文件清单

| 文件 | 操作 | 备注 |
|------|------|------|
| `src/entropy/__init__.py` | 新建 | 模块入口 |
| `src/entropy/scanner.py` | 新建 | 扫描器核心 |
| `src/entropy/detectors/doc_drift.py` | 新建 | 文档漂移检测 |
| `src/entropy/detectors/style_drift.py` | 新建 | 风格漂移检测 |
| `src/entropy/detectors/dead_code.py` | 新建 | 死代码检测 |
| `src/entropy/detectors/constraint.py` | 新建 | 约束违反检测 |
| `src/entropy/detectors/artifact.py` | 新建 | 产物质量检测 |
| `src/entropy/cleaners/doc_cleaner.py` | 新建 | 文档清理 |
| `src/entropy/cleaners/style_cleaner.py` | 新建 | 风格清理 |
| `src/entropy/cleaners/dead_code_cleaner.py` | 新建 | 死代码清理 |
| `src/entropy/scheduler.py` | 新建 | 调度器 |
| `src/entropy/report.py` | 新建 | 报告生成 |
| `src/entropy/hooks.py` | 新建 | CI/Pre-commit 集成 |
| `src/entropy/cli.py` | 新建 | CLI 入口 |
| `src/api/routes/entropy.py` | 新建 | API 端点 |
| `frontend/src/components/EntropyDashboard.tsx` | 新建 | 前端 Dashboard |
| `.pre-commit-config.yaml` | 新建/修改 | Pre-commit Hook |
| `.github/workflows/entropy.yml` | 新建 | CI 集成 |
| `tests/entropy/test_scanner.py` | 新建 | 扫描器测试 |
| `tests/entropy/test_detectors.py` | 新建 | 检测器测试 |

---

## 十三、参考

- [Harness Engineering: The Complete Guide — NxCode](https://www.nxcode.io/resources/news/harness-engineering-complete-guide-ai-agent-codex-2026)
- [LangChain Coding Agent Harness — LangChain Blog](https://blog.langchain.com)
- [pre-commit framework](https://pre-commit.com)
- [ArchUnit](https://www.archunit.org) — Java 架构约束工具（参考设计理念）
- [pygrep-hook](https://pre-commit.com/#python) — Python linter hook 模式
