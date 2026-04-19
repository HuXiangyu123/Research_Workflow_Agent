/**
 * Phase 3-4 types — ReviewFeedback, Trace, Workspace, Agent, Skills, MCP, Config
 */

// ─── Phase 3: Review ────────────────────────────────────────────────────────────

export type ReviewSeverity = 'info' | 'warning' | 'error' | 'blocker';
export type ReviewCategory =
  | 'coverage_gap'
  | 'unsupported_claim'
  | 'citation_reachability'
  | 'duplication'
  | 'consistency';
export type RevisionActionType =
  | 'research_more'
  | 'rewrite_section'
  | 'fix_citation'
  | 'drop_claim'
  | 'merge_duplicate';

export interface CoverageGap {
  sub_question_id?: string;
  missing_topics: string[];
  missing_papers: string[];
  note?: string;
}

export interface ClaimSupport {
  claim_id: string;
  claim_text: string;
  supported: boolean;
  evidence_chunk_ids: string[];
  citation_ids: string[];
  note?: string;
}

export interface RevisionAction {
  action_type: RevisionActionType;
  target: string;
  reason: string;
  priority: number;
}

export interface ReviewIssue {
  issue_id: string;
  severity: ReviewSeverity;
  category: ReviewCategory;
  target: string;
  summary: string;
  evidence_refs: string[];
}

export interface ReviewFeedback {
  schema_version?: string;
  review_id: string;
  task_id: string;
  workspace_id: string;
  passed: boolean;
  issues: ReviewIssue[];
  coverage_gaps: CoverageGap[];
  claim_supports: ClaimSupport[];
  revision_actions: RevisionAction[];
  summary?: string;
  created_at?: string;
}

// ─── Phase 3: Trace ─────────────────────────────────────────────────────────────

export type RunStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';

export interface NodeRun {
  run_id: string;
  task_id: string;
  workspace_id: string;
  node_name: string;
  stage: string;
  status: RunStatus;
  started_at?: string;
  ended_at?: string;
  input_artifact_ids: string[];
  output_artifact_ids: string[];
  warning_messages: string[];
  error_message?: string;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
}

export interface ToolRun {
  tool_run_id: string;
  parent_run_id: string;
  task_id: string;
  workspace_id: string;
  node_name: string;
  tool_name: string;
  status: RunStatus;
  started_at?: string;
  ended_at?: string;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
  error_message?: string;
  duration_ms?: number;
}

export type TraceEventType =
  | 'task_created'
  | 'node_started'
  | 'node_finished'
  | 'node_failed'
  | 'tool_started'
  | 'tool_finished'
  | 'tool_failed'
  | 'artifact_saved'
  | 'review_generated'
  | 'warning'
  | 'task_completed'
  | 'task_failed'
  | 'stage_changed';

export interface TraceEvent {
  event_id: string;
  task_id: string;
  workspace_id: string;
  run_id?: string;
  tool_run_id?: string;
  event_type: TraceEventType;
  ts: string;
  payload: Record<string, unknown>;
}

export interface TaskTraceResponse {
  task_id: string;
  node_runs: NodeRun[];
  tool_runs: ToolRun[];
  events: TraceEvent[];
}

// ─── Phase 3: Workspace ─────────────────────────────────────────────────────────

export type ArtifactType =
  | 'brief'
  | 'search_plan'
  | 'paper_card'
  | 'rag_result'
  | 'comparison_matrix'
  | 'report_outline'
  | 'report_draft'
  | 'review_feedback'
  | 'node_trace'
  | 'tool_trace'
  | 'eval_report'
  | 'upload'
  | 'task_log'
  | 'raw_input';

export interface ArtifactRef {
  artifact_id: string;
  artifact_type: ArtifactType;
  title: string;
}

export interface WorkspaceArtifact {
  artifact_id: string;
  workspace_id: string;
  task_id?: string;
  artifact_type: ArtifactType;
  title: string;
  status: string;
  created_at: string;
  created_by_node?: string;
  content_ref?: string | null;  // 后端默认 None，可能返回 null
  summary?: string;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface WorkspaceSummary {
  workspace_id: string;
  status: string;
  current_stage?: string;
  warnings: string[];
  artifact_count: number;
}

// ─── Phase 4: Config ────────────────────────────────────────────────────────────

export type ExecutionMode = 'legacy' | 'hybrid' | 'v2';
export type AgentMode = 'auto' | 'planner' | 'retriever' | 'analyst' | 'reviewer';
export type SupervisorMode = 'graph' | 'llm_handoff';

export interface Phase4Config {
  execution_mode: ExecutionMode;
  agent_mode: AgentMode;
  supervisor_mode: SupervisorMode;
  enable_mcp: boolean;
  enable_skills: boolean;
  enable_replan: boolean;
  auto_fill: boolean;
  node_backends: NodeBackendConfig;
}

export interface NodeBackendConfig {
  clarify: 'legacy' | 'v2' | 'auto';
  search_plan: 'legacy' | 'v2' | 'auto';
  search: 'legacy' | 'v2' | 'auto';
  extract: 'legacy' | 'v2' | 'auto';
  draft: 'legacy' | 'v2' | 'auto';
  review: 'legacy' | 'v2' | 'auto';
  persist_artifacts: 'legacy' | 'v2' | 'auto';
  plan_search?: 'legacy' | 'v2' | 'auto';
  search_corpus?: 'legacy' | 'v2' | 'auto';
  extract_cards?: 'legacy' | 'v2' | 'auto';
  synthesize?: 'legacy' | 'v2' | 'auto';
  revise?: 'legacy' | 'v2' | 'auto';
  write_report?: 'legacy' | 'v2' | 'auto';
}

// ─── Phase 4: Agent ───────────────────────────────────────────────────────────────

export type AgentRole = 'supervisor' | 'planner' | 'retriever' | 'analyst' | 'reviewer';
export type AgentVisibility = 'auto' | 'explicit' | 'both';

export interface AgentDescriptor {
  agent_id: string;
  role: AgentRole;
  title: string;
  description: string;
  visibility: AgentVisibility;
  supported_skills: string[];
  supported_nodes: string[];
}

// 后端 ReplanResponse 的响应形状
export interface ReplanResponse {
  replan_id: string;
  workspace_id: string;
  task_id: string;
  trigger: 'reviewer' | 'retriever' | 'user';
  target_stage: string;
  output_artifact_ids: string[];
  trace_refs: string[];
  collaboration_trace?: Array<Record<string, unknown>>;
  summary?: string;
}

// ─── Phase 4: Skills ────────────────────────────────────────────────────────────

export type SkillBackend =
  | 'local_graph'
  | 'local_function'
  | 'mcp_prompt'
  | 'mcp_toolchain';
export type SkillVisibility = 'auto' | 'explicit' | 'both';

export interface SkillMeta {
  skill_id: string;
  name: string;
  description: string;
  backend: SkillBackend;
  default_agent: AgentRole;
  tags: string[];
  visibility: SkillVisibility;
}

export interface SkillManifest extends SkillMeta {
  input_schema: Record<string, unknown>;
  output_artifact_type?: string;
  backend_ref: string;
}

// ─── Phase 4: MCP ────────────────────────────────────────────────────────────────

export type MCPServerTransport = 'stdio' | 'remote';
export type MCPInvokeKind = 'tool' | 'prompt' | 'resource';

export interface MCPServerConfig {
  server_id: string;
  name: string;
  transport: MCPServerTransport;
  command?: string;
  args?: string[];
  url?: string;
  env?: Record<string, string>;
  enabled: boolean;
  workspace_scoped: boolean;
  auth_ref?: string;   // 后端 MCPServerConfig.model 有此字段
}

export interface MCPToolDescriptor {
  server_id: string;
  tool_name: string;
  title: string;
  description: string;
  input_schema: Record<string, unknown>;
  requires_approval: boolean;
  tags: string[];
}

export interface MCPCatalog {
  tools: MCPToolDescriptor[];
  prompts: unknown[];
  resources: unknown[];
}

// ─── Phase 3-4 API responses ─────────────────────────────────────────────────────

export interface TaskReviewResponse {
  task_id: string;
  review_feedback: ReviewFeedback | null;
  review_passed: boolean | null;
}

export type Phase3_4Tab = 'config' | 'agents' | 'skills' | 'review' | 'trace';
