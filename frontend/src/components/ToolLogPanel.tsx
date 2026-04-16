import { useRef, useEffect } from 'react';
import type { TaskEvent } from '../types/task';

interface Props {
  events: TaskEvent[];
}

export function ToolLogPanel({ events }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  return (
    <div className="h-full overflow-y-auto rounded-xl border border-stone-300 bg-white p-3 font-mono text-[11px] leading-relaxed shadow-sm">
      {events.length === 0 && (
        <p className="text-stone-400 italic">Waiting for events…</p>
      )}
      {events.map((ev, i) => (
        <div key={i} className="py-0.5 border-b border-stone-100 last:border-0">
          {ev.type === 'node_start' && (
            <span className="text-[#1e40af]">▶ {ev.node} started</span>
          )}
          {ev.type === 'node_end' && (
            <span className={ev.status === 'failed' ? 'text-[#b91c1c]' : 'text-[#166534]'}>
              {ev.status === 'failed' ? '✗' : '✓'} {ev.node} {ev.status === 'failed' ? 'failed' : 'done'}
              {ev.duration_ms
                ? ` (${ev.duration_ms >= 1000 ? (ev.duration_ms / 1000).toFixed(1) + 's' : ev.duration_ms + 'ms'})`
                : ''}
              {ev.tokens_delta ? ` ${ev.tokens_delta} tok` : ''}
            </span>
          )}
          {ev.type === 'thinking' && (
            <span className="text-[#7c3aed]">
              💭 {ev.node}: {ev.content?.slice(0, 80) ?? ''}
              {(ev.content?.length ?? 0) > 80 ? '…' : ''}
            </span>
          )}
          {ev.type === 'status_change' && (
            <span className="text-[#a16207]">⚡ status → {ev.status}</span>
          )}
          {ev.type === 'artifact_ready' && (
            <span className="text-[#0f766e]">
              ◇ artifact ready → {ev.artifact_name}
              {ev.summary ? ` (${ev.summary})` : ''}
            </span>
          )}
          {ev.type === 'report_snapshot' && (
            <span className="text-[#7c2d12]">
              ✎ live markdown → {ev.artifact_name}
              {ev.is_final ? ' (final)' : ''}
            </span>
          )}
          {ev.type === 'done' && (
            <span
              className={
                ev.status === 'completed' ? 'text-[#166534] font-medium' : 'text-[#b91c1c] font-medium'
              }
            >
              ■ Task {ev.status}
            </span>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
