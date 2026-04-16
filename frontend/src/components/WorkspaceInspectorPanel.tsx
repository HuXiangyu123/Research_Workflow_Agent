import { useState } from 'react';
import type { TaskEvent } from '../types/task';
import { SkillPalette } from './SkillPalette';
import { ToolLogPanel } from './ToolLogPanel';
import { WorkspaceArtifactsPanel } from './WorkspaceArtifactsPanel';

type InspectorTab = 'artifacts' | 'skills' | 'events';

interface Props {
  taskId?: string | null;
  workspaceId?: string | null;
  isRunning?: boolean;
  events: TaskEvent[];
  highlightArtifact?: string | null;
}

const TABS: Array<{ id: InspectorTab; label: string }> = [
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'skills', label: 'Skills' },
  { id: 'events', label: 'Events' },
];

export function WorkspaceInspectorPanel({
  taskId,
  workspaceId,
  isRunning = false,
  events,
  highlightArtifact,
}: Props) {
  const [tab, setTab] = useState<InspectorTab>('artifacts');

  return (
    <div className="h-full overflow-hidden rounded-xl border border-stone-300 bg-[#faf8f5] shadow-sm">
      <div className="border-b border-stone-200 bg-white px-3 py-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-stone-500">Inspector</p>
            <p className="mt-1 text-[11px] text-stone-400">
              {workspaceId ? <span className="font-mono">{workspaceId}</span> : 'No active workspace'}
            </p>
          </div>
          {isRunning && (
            <span className="rounded-full bg-[#eff6ff] px-2.5 py-1 text-[10px] font-medium text-[#1e40af]">
              live
            </span>
          )}
        </div>
        <div className="mt-3 flex gap-1">
          {TABS.map(item => (
            <button
              key={item.id}
              onClick={() => setTab(item.id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                tab === item.id
                  ? 'bg-[#1e3a5f] text-white'
                  : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="h-[610px] p-3">
        {tab === 'artifacts' && (
          <WorkspaceArtifactsPanel
            workspaceId={workspaceId}
            isRunning={isRunning}
            highlightArtifact={highlightArtifact}
          />
        )}
        {tab === 'skills' && (
          <div className="h-full overflow-y-auto rounded-xl border border-stone-300 bg-white p-3 shadow-sm">
            <SkillPalette taskId={taskId} workspaceId={workspaceId ?? undefined} />
          </div>
        )}
        {tab === 'events' && <ToolLogPanel events={events} />}
      </div>
    </div>
  );
}
