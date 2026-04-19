export type SourceType = 'arxiv' | 'pdf' | 'research';
export type ReportMode = 'draft' | 'full';
export type PaperType = 'regular' | 'survey';
export type WorkflowMode = 'report' | 'research';

export interface TaskEvent {
  type:
    | 'node_start'
    | 'node_end'
    | 'status_change'
    | 'done'
    | 'thinking'
    | 'artifact_ready'
    | 'report_snapshot';
  node?: string;
  status?: string;
  timestamp?: string;
  tokens_delta?: number;
  warnings?: string[];
  duration_ms?: number;
  content?: string;
  error?: string;
  artifact_name?: string;
  workspace_id?: string;
  summary?: string;
  is_final?: boolean;
}

export interface ThinkingEntry {
  node: string;
  content: string;
}

export interface Task {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  report_mode?: ReportMode;
  source_type?: SourceType;
  paper_type?: PaperType;
  draft_markdown?: string;
  full_markdown?: string;
  result_markdown?: string;
  workspace_id?: string;
  followup_hints?: string[];
  chat_history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  error?: string;
  current_stage?: string;
  // research 模式下完整 state
  brief?: ResearchBrief;
  search_plan?: SearchPlan;
  rag_result?: unknown;
  paper_cards?: Array<Record<string, unknown>>;
  compression_result?: unknown;
  taxonomy?: unknown;
  draft_report?: unknown;
  review_feedback?: unknown;
  review_passed?: boolean;
  artifacts_created?: Array<Record<string, unknown>>;
  artifact_count?: number;
  collaboration_trace?: Array<Record<string, unknown>>;
  supervisor_mode?: 'graph' | 'llm_handoff' | string;
  awaiting_followup?: boolean;
  followup_resolution?: unknown;
  persisted_to_db?: boolean;
  persisted_report_id?: string;
  persistence_error?: string;
}

export interface ResearchAmbiguity {
  field: string;
  reason: string;
  suggested_options?: string[];
}

export interface ResearchBrief {
  topic: string;
  goal: string;
  desired_output?: string;
  sub_questions?: string[];
  time_range?: string;
  domain_scope?: string;
  source_constraints?: string[];
  focus_dimensions?: string[];
  ambiguities?: ResearchAmbiguity[];
  needs_followup?: boolean;
  confidence?: number;
  schema_version?: string;
}

export interface SearchPlan {
  schema_version?: string;
  plan_goal: string;
  coverage_strategy?: 'broad' | 'focused' | 'hybrid';
  query_groups: SearchQueryGroup[];
  source_preferences?: string[];
  dedup_strategy?: 'exact' | 'semantic' | 'none';
  rerank_required?: boolean;
  max_candidates_per_query?: number;
  requires_local_corpus?: boolean;
  coverage_notes?: string;
  planner_warnings?: string[];
  followup_search_seeds?: string[];
  followup_needed?: boolean;
}

export interface SearchQueryGroup {
  group_id: string;
  queries: string[];
  intent?: string;
  priority?: number;
  expected_hits?: number;
  notes?: string;
}

// ─── Graph 节点列表 ────────────────────────────────────────────────────────────

export const REPORT_GRAPH_NODES = [
  'input_parse',
  'ingest_source',
  'extract_document_text',
  'normalize_metadata',
  'retrieve_evidence',
  'classify_paper_type',
  'draft_report',
  'report_frame',
  'survey_intro_outline',
  'repair_report',
  'resolve_citations',
  'verify_claims',
  'apply_policy',
  'format_output',
] as const;

export const RESEARCH_GRAPH_NODES = [
  'clarify',
  'search_plan',
  'search',
  'extract',
  'extract_compression',
  'draft',
  'review',
  'persist_artifacts',
] as const;

export type ReportNodeName = typeof REPORT_GRAPH_NODES[number];
export type ResearchNodeName = typeof RESEARCH_GRAPH_NODES[number];

export type AnyNodeName = ReportNodeName | ResearchNodeName;
export type NodeName = AnyNodeName;
export type NodeStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped';

export const GRAPH_NODES = [
  ...REPORT_GRAPH_NODES,
  ...RESEARCH_GRAPH_NODES,
] as const satisfies readonly AnyNodeName[];

export const GRAPH_NODE_LABELS: Record<AnyNodeName, string> = {
  input_parse: 'Input Parse',
  ingest_source: 'Ingest Source',
  extract_document_text: 'Extract Text',
  normalize_metadata: 'Normalize',
  retrieve_evidence: 'Retrieve Evidence',
  classify_paper_type: 'Classify Paper',
  draft_report: 'Draft Report',
  report_frame: 'Full Report',
  survey_intro_outline: 'Survey Intro+Outline',
  repair_report: 'Repair Report',
  resolve_citations: 'Resolve Citations',
  verify_claims: 'Verify Claims',
  apply_policy: 'Apply Policy',
  format_output: 'Format Output',
  clarify: 'Clarify Brief',
  search_plan: 'Search Plan',
  search: 'Search Papers',
  extract: 'Extract Cards',
  extract_compression: 'Compress Evidence',
  draft: 'Draft Survey',
  review: 'Review',
  persist_artifacts: 'Persist',
};

export function getWorkflowMode(sourceType?: SourceType | null): WorkflowMode {
  return sourceType === 'research' ? 'research' : 'report';
}

export function getGraphNodes(sourceType?: SourceType | null): readonly AnyNodeName[] {
  return getWorkflowMode(sourceType) === 'research' ? RESEARCH_GRAPH_NODES : REPORT_GRAPH_NODES;
}

export function getNodeLabel(node: string): string {
  return GRAPH_NODE_LABELS[node as AnyNodeName] ?? node;
}
