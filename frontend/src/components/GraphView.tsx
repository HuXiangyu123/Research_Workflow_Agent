import { useMemo } from 'react';
import { Background, Controls, ReactFlow, type Edge, type Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  GRAPH_NODE_LABELS,
  getWorkflowMode,
  type AnyNodeName,
  type NodeStatus,
  type SourceType,
} from '../types/task';

interface Props {
  nodeStatuses: Record<AnyNodeName, NodeStatus>;
  sourceType?: SourceType | null;
}

const STATUS_STYLES: Record<
  NodeStatus,
  { bg: string; color: string; border: string; shadow?: string }
> = {
  pending: {
    bg: '#e7e5e4',
    color: '#44403c',
    border: '#d6d3d1',
  },
  running: {
    bg: '#1e40af',
    color: '#ffffff',
    border: '#1d4ed8',
    shadow: '0 0 0 3px rgba(30, 64, 175, 0.2)',
  },
  done: {
    bg: '#166534',
    color: '#ffffff',
    border: '#15803d',
  },
  failed: {
    bg: '#b91c1c',
    color: '#ffffff',
    border: '#991b1b',
  },
  skipped: {
    bg: '#a8a29e',
    color: '#fafaf9',
    border: '#78716c',
  },
};

type LayoutEntry = {
  id: AnyNodeName;
  x: number;
  y: number;
};

type EdgeDef = {
  source: AnyNodeName;
  target: AnyNodeName;
  label?: string;
};

const COL_LEFT = 20;
const COL_CENTER = 200;
const COL_RIGHT = 380;
const ROW_H = 64;

const REPORT_LAYOUT: LayoutEntry[] = [
  { id: 'input_parse', x: COL_CENTER, y: 0 * ROW_H },
  { id: 'ingest_source', x: COL_CENTER, y: 1 * ROW_H },
  { id: 'extract_document_text', x: COL_CENTER, y: 2 * ROW_H },
  { id: 'normalize_metadata', x: COL_CENTER, y: 3 * ROW_H },
  { id: 'retrieve_evidence', x: COL_CENTER, y: 4 * ROW_H },
  { id: 'classify_paper_type', x: COL_CENTER, y: 5 * ROW_H },
  { id: 'draft_report', x: COL_LEFT, y: 6.2 * ROW_H },
  { id: 'report_frame', x: COL_CENTER, y: 6.2 * ROW_H },
  { id: 'survey_intro_outline', x: COL_RIGHT, y: 6.2 * ROW_H },
  { id: 'repair_report', x: COL_LEFT, y: 7.4 * ROW_H },
  { id: 'resolve_citations', x: COL_CENTER, y: 8.6 * ROW_H },
  { id: 'verify_claims', x: COL_CENTER, y: 9.6 * ROW_H },
  { id: 'apply_policy', x: COL_CENTER, y: 10.6 * ROW_H },
  { id: 'format_output', x: COL_CENTER, y: 11.6 * ROW_H },
];

const REPORT_EDGE_DEFS: EdgeDef[] = [
  { source: 'input_parse', target: 'ingest_source' },
  { source: 'ingest_source', target: 'extract_document_text' },
  { source: 'extract_document_text', target: 'normalize_metadata' },
  { source: 'normalize_metadata', target: 'retrieve_evidence' },
  { source: 'retrieve_evidence', target: 'classify_paper_type' },
  { source: 'classify_paper_type', target: 'draft_report', label: 'draft' },
  { source: 'classify_paper_type', target: 'report_frame', label: 'regular+full' },
  { source: 'classify_paper_type', target: 'survey_intro_outline', label: 'survey+full' },
  { source: 'draft_report', target: 'repair_report' },
  { source: 'repair_report', target: 'resolve_citations' },
  { source: 'report_frame', target: 'resolve_citations' },
  { source: 'survey_intro_outline', target: 'resolve_citations' },
  { source: 'resolve_citations', target: 'verify_claims' },
  { source: 'verify_claims', target: 'apply_policy' },
  { source: 'apply_policy', target: 'format_output' },
];

const RESEARCH_LAYOUT: LayoutEntry[] = [
  { id: 'clarify', x: COL_CENTER, y: 0 * ROW_H },
  { id: 'search_plan', x: COL_CENTER, y: 1.2 * ROW_H },
  { id: 'search', x: COL_CENTER, y: 2.4 * ROW_H },
  { id: 'extract', x: COL_CENTER, y: 3.6 * ROW_H },
  { id: 'extract_compression', x: COL_CENTER, y: 4.8 * ROW_H },
  { id: 'draft', x: COL_CENTER, y: 6 * ROW_H },
  { id: 'review', x: COL_CENTER, y: 7.2 * ROW_H },
  { id: 'persist_artifacts', x: COL_CENTER, y: 8.4 * ROW_H },
];

const RESEARCH_EDGE_DEFS: EdgeDef[] = [
  { source: 'clarify', target: 'search_plan', label: 'if no followup' },
  { source: 'search_plan', target: 'search', label: 'full research' },
  { source: 'search', target: 'extract' },
  { source: 'extract', target: 'extract_compression' },
  { source: 'extract_compression', target: 'draft' },
  { source: 'draft', target: 'review' },
  { source: 'review', target: 'persist_artifacts', label: 'if passed' },
];

function makeNodeStyle(st: (typeof STATUS_STYLES)[NodeStatus], isRunning: boolean) {
  return {
    background: st.bg,
    color: st.color,
    border: `1px solid ${st.border}`,
    borderRadius: '8px',
    padding: '6px 14px',
    fontSize: '11px',
    fontWeight: isRunning ? 600 : 500,
    minWidth: '140px',
    textAlign: 'center' as const,
    boxShadow: st.shadow ?? 'none',
    fontFamily: '"IBM Plex Sans", system-ui, sans-serif',
  };
}

export function GraphView({ nodeStatuses, sourceType }: Props) {
  const workflowMode = getWorkflowMode(sourceType);

  const { nodes, edges } = useMemo(() => {
    const layout = workflowMode === 'research' ? RESEARCH_LAYOUT : REPORT_LAYOUT;
    const edgeDefs = workflowMode === 'research' ? RESEARCH_EDGE_DEFS : REPORT_EDGE_DEFS;

    const nodes: Node[] = layout.map(({ id, x, y }) => {
      const status = nodeStatuses[id] ?? 'pending';
      const st = STATUS_STYLES[status];
      return {
        id,
        position: { x, y },
        data: { label: GRAPH_NODE_LABELS[id] },
        style: makeNodeStyle(st, status === 'running'),
      };
    });

    const edges: Edge[] = edgeDefs.map(({ source, target, label }) => ({
      id: `${source}-${target}`,
      source,
      target,
      animated: nodeStatuses[source] === 'running',
      label,
      labelStyle: { fontSize: 9, fill: '#78716c' },
      labelBgStyle: { fill: '#faf8f5', fillOpacity: 0.85 },
      labelBgPadding: [4, 2] as [number, number],
      style: { stroke: '#a8a29e', strokeWidth: 1.5 },
    }));

    return { nodes, edges };
  }, [nodeStatuses, workflowMode]);

  return (
    <div className="h-full w-full bg-[#faf8f5]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
      >
        <Background color="#d6d3d1" gap={20} size={1} />
        <Controls
          showInteractive={false}
          className="!bg-white !border-stone-300 !shadow-md [&_button]:!fill-stone-700"
        />
      </ReactFlow>
    </div>
  );
}
