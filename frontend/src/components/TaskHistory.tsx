import { useEffect, useState } from 'react';
import type { SourceType } from '../types/task';

interface WorkspaceSummary {
  workspace_id: string;
  latest_task_id?: string | null;
  status: string;
  updated_at?: string | null;
  current_stage?: string | null;
  source_type?: SourceType;
  report_mode?: string | null;
}

interface Props {
  onSelect: (taskId: string, sourceType?: SourceType) => void;
  refreshTrigger: number;
}

export function TaskHistory({ onSelect, refreshTrigger }: Props) {
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);

  useEffect(() => {
    fetch('/api/v1/workspaces')
      .then(r => r.json())
      .then(data => setWorkspaces(data.items ?? []))
      .catch(() => {});
  }, [refreshTrigger]);

  if (workspaces.length === 0) return null;

  return (
    <div className="space-y-1">
      <h3 className="text-xs font-semibold text-stone-500 uppercase tracking-wider px-1">
        Workspaces
      </h3>
      {workspaces.map(workspace => {
        const latestTaskId = workspace.latest_task_id ?? undefined;
        const disabled = !latestTaskId;
        return (
        <button
          key={workspace.workspace_id}
          onClick={() => latestTaskId && onSelect(latestTaskId, workspace.source_type)}
          disabled={disabled}
          className={`w-full text-left px-3 py-2 rounded-lg border border-transparent transition-all text-sm ${
            disabled
              ? 'cursor-not-allowed opacity-60'
              : 'hover:bg-white hover:border-stone-200 hover:shadow-sm'
          }`}
        >
          <span className="text-stone-800 font-mono text-xs">{workspace.workspace_id.slice(0, 22)}…</span>
          {workspace.source_type && (
            <span className="ml-2 rounded-full bg-stone-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-stone-600">
              {workspace.source_type === 'research' ? 'research' : 'report'}
            </span>
          )}
          <span
            className={`ml-2 text-xs font-medium ${
              workspace.status === 'completed'
                ? 'text-[#166534]'
                : workspace.status === 'failed'
                  ? 'text-[#b91c1c]'
                  : workspace.status === 'running'
                    ? 'text-[#1e40af]'
                    : 'text-stone-500'
            }`}
          >
            {workspace.status}
          </span>
          {workspace.current_stage && (
            <p className="mt-1 text-[11px] text-stone-500">stage: {workspace.current_stage}</p>
          )}
          {latestTaskId && (
            <p className="mt-1 font-mono text-[10px] text-stone-400">task {latestTaskId.slice(0, 8)}…</p>
          )}
        </button>
        );
      })}
    </div>
  );
}
