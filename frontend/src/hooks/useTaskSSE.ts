import { useState, useEffect, useRef } from 'react';
import type { AnyNodeName, NodeStatus, SourceType, Task, TaskEvent, ThinkingEntry } from '../types/task';
import { GRAPH_NODES } from '../types/task';

interface SSEState {
  nodeStatuses: Record<AnyNodeName, NodeStatus>;
  events: TaskEvent[];
  thinkingEntries: ThinkingEntry[];
  totalDurationMs: number;
  isDone: boolean;
  error: string | null;
  sourceType: SourceType | null;
  currentStage: string | null;
  taskStatus: Task['status'] | null;
  workspaceId: string | null;
  latestReportMarkdown: string | null;
  latestReportArtifact: string | null;
}

const INITIAL_STATUSES = () =>
  Object.fromEntries(GRAPH_NODES.map(node => [node, 'pending'])) as Record<AnyNodeName, NodeStatus>;

function createInitialState(): SSEState {
  return {
    nodeStatuses: INITIAL_STATUSES(),
    events: [],
    thinkingEntries: [],
    totalDurationMs: 0,
    isDone: false,
    error: null,
    sourceType: null,
    currentStage: null,
    taskStatus: null,
    workspaceId: null,
    latestReportMarkdown: null,
    latestReportArtifact: null,
  };
}

function isKnownNode(node?: string): node is AnyNodeName {
  return Boolean(node && GRAPH_NODES.includes(node as AnyNodeName));
}

export function useTaskSSE(taskId: string | null) {
  const [state, setState] = useState<SSEState>(createInitialState);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!taskId) {
      setState(createInitialState());
      return;
    }

    let isMounted = true;
    setState(createInitialState());

    fetch(`/tasks/${taskId}`)
      .then(r => r.json())
      .then((task: Task) => {
        if (!isMounted) return;
        setState(prev => ({
          ...prev,
          sourceType: task.source_type ?? null,
          currentStage: task.current_stage ?? null,
          taskStatus: task.status,
          workspaceId: task.workspace_id ?? null,
          isDone: task.status === 'completed' || task.status === 'failed',
        }));
      })
      .catch(() => {
        if (!isMounted) return;
        setState(prev => ({ ...prev, error: 'Failed to load task metadata' }));
      });

    const es = new EventSource(`/tasks/${taskId}/events`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const event: TaskEvent = JSON.parse(e.data);
        setState(prev => {
          const newStatuses = { ...prev.nodeStatuses };
          let newThinking = prev.thinkingEntries;
          let newDuration = prev.totalDurationMs;
          let currentStage = prev.currentStage;
          let taskStatus = prev.taskStatus;
          let workspaceId = prev.workspaceId;
          let latestReportMarkdown = prev.latestReportMarkdown;
          let latestReportArtifact = prev.latestReportArtifact;

          if (event.type === 'node_start' && isKnownNode(event.node)) {
            newStatuses[event.node] = 'running';
            currentStage = event.node;
            taskStatus = 'running';
          } else if (event.type === 'node_end' && isKnownNode(event.node)) {
            newStatuses[event.node] =
              event.status === 'failed'
                ? 'failed'
                : event.status === 'skipped'
                  ? 'skipped'
                  : 'done';
            currentStage = event.node;
            if (event.duration_ms) {
              newDuration += event.duration_ms;
            }
          } else if (event.type === 'thinking' && event.node && event.content) {
            newThinking = [...prev.thinkingEntries, { node: event.node, content: event.content }];
          } else if (event.type === 'status_change' && event.status) {
            taskStatus = event.status as Task['status'];
          } else if (event.type === 'report_snapshot' && event.content) {
            latestReportMarkdown = event.content;
            latestReportArtifact = event.artifact_name ?? null;
          }

          if (event.workspace_id) {
            workspaceId = event.workspace_id;
          }

          const isDone = event.type === 'done';
          if (event.type === 'done' && event.status) {
            taskStatus = event.status as Task['status'];
          }
          return {
            nodeStatuses: newStatuses,
            events: [...prev.events, event],
            thinkingEntries: newThinking,
            totalDurationMs: newDuration,
            isDone,
            error: event.type === 'done' && event.status === 'failed' ? 'Task failed' : prev.error,
            sourceType: prev.sourceType,
            currentStage,
            taskStatus,
            workspaceId,
            latestReportMarkdown,
            latestReportArtifact,
          };
        });
        if (event.type === 'done') {
          es.close();
        }
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setState(prev => ({ ...prev, error: 'SSE connection lost', isDone: true }));
      es.close();
    };

    return () => {
      isMounted = false;
      es.close();
      esRef.current = null;
    };
  }, [taskId]);

  return state;
}
