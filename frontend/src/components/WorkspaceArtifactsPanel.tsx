import { useEffect, useMemo, useState } from 'react';
import type { WorkspaceArtifact } from '../types/phase34';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ArtifactContent {
  workspace_id: string;
  artifact_id: string;
  artifact_type: string;
  title: string;
  content_type: 'json' | 'markdown' | 'text';
  content: unknown;
  path: string;
}

interface Props {
  workspaceId?: string | null;
  isRunning?: boolean;
  highlightArtifact?: string | null;
}

export function WorkspaceArtifactsPanel({
  workspaceId,
  isRunning = false,
  highlightArtifact,
}: Props) {
  const [items, setItems] = useState<WorkspaceArtifact[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [content, setContent] = useState<ArtifactContent | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!workspaceId) {
      setItems([]);
      setSelectedId(null);
      setContent(null);
      return;
    }

    let cancelled = false;

    const load = () => {
      fetch(`/api/v1/workspaces/${workspaceId}/artifacts`)
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then((data: { items?: WorkspaceArtifact[] }) => {
          if (cancelled) return;
          const nextItems = data.items ?? [];
          setItems(nextItems);
          setSelectedId(prev => {
            if (highlightArtifact) {
              const highlighted = nextItems.find(item => item.content_ref?.endsWith(highlightArtifact));
              if (highlighted) return highlighted.artifact_id;
            }
            if (prev && nextItems.some(item => item.artifact_id === prev)) return prev;
            return nextItems[0]?.artifact_id ?? null;
          });
        })
        .catch(() => {
          if (!cancelled) {
            setItems([]);
          }
        });
    };

    load();
    if (!isRunning) {
      return () => {
        cancelled = true;
      };
    }

    const timer = window.setInterval(load, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [workspaceId, isRunning, highlightArtifact]);

  useEffect(() => {
    if (!workspaceId || !selectedId) {
      setContent(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    fetch(`/api/v1/workspaces/${workspaceId}/artifacts/${selectedId}/content`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ArtifactContent) => {
        if (!cancelled) {
          setContent(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setContent(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceId, selectedId]);

  const selectedItem = useMemo(
    () => items.find(item => item.artifact_id === selectedId) ?? null,
    [items, selectedId],
  );

  if (!workspaceId) {
    return (
      <div className="h-full rounded-xl border border-stone-300 bg-white p-4 text-sm text-stone-500 shadow-sm">
        Workspace artifacts will appear here once a workspace is selected.
      </div>
    );
  }

  return (
    <div className="h-full overflow-hidden rounded-xl border border-stone-300 bg-white shadow-sm">
      <div className="border-b border-stone-200 px-4 py-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-stone-500">Workspace Outputs</p>
        <p className="mt-1 font-mono text-[11px] text-stone-600">{workspaceId}</p>
      </div>

      <div className="grid h-[610px] grid-rows-[220px_minmax(0,1fr)]">
        <div className="overflow-y-auto border-b border-stone-200 p-3">
          {items.length === 0 ? (
            <p className="text-sm text-stone-500">
              {isRunning ? 'Waiting for workspace artifacts…' : 'No artifacts found.'}
            </p>
          ) : (
            <div className="space-y-2">
              {items.map(item => {
                const active = item.artifact_id === selectedId;
                return (
                  <button
                    key={item.artifact_id}
                    onClick={() => setSelectedId(item.artifact_id)}
                    className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                      active
                        ? 'border-[#1e3a5f] bg-[#eff6ff]'
                        : 'border-stone-200 bg-stone-50 hover:bg-white'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="rounded-full bg-stone-200 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-stone-700">
                        {item.artifact_type}
                      </span>
                      <span className="text-[10px] text-stone-400">{new Date(item.created_at).toLocaleTimeString()}</span>
                    </div>
                    <p className="mt-1 text-sm font-medium text-stone-800">{item.title}</p>
                    {item.summary && (
                      <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-stone-500">{item.summary}</p>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="overflow-y-auto p-4">
          {!selectedItem ? (
            <p className="text-sm text-stone-500">Select an artifact to inspect its content.</p>
          ) : loading ? (
            <p className="text-sm text-stone-500">Loading artifact content…</p>
          ) : !content ? (
            <p className="text-sm text-stone-500">Artifact content is unavailable.</p>
          ) : (
            <div className="space-y-3">
              <div>
                <p className="text-sm font-semibold text-stone-800">{content.title}</p>
                <p className="mt-1 text-[11px] text-stone-400">{content.path}</p>
              </div>
              {content.content_type === 'markdown' && typeof content.content === 'string' ? (
                <MarkdownRenderer content={content.content} />
              ) : (
                <pre className="overflow-x-auto rounded-lg bg-stone-50 p-3 text-[11px] leading-relaxed text-stone-700">
                  {typeof content.content === 'string'
                    ? content.content
                    : JSON.stringify(content.content, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
