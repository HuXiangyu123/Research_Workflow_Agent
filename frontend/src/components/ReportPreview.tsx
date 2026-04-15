import { useEffect, useState } from 'react';
import type { ResearchBrief, SearchPlan, SearchQueryGroup, SourceType, Task } from '../types/task';
import { MarkdownRenderer } from './MarkdownRenderer';
import { ResearchFollowupForm } from './ResearchFollowupForm';

interface Props {
  taskId: string | null;
  isDone: boolean;
  sourceType?: SourceType | null;
  onTaskCreated: (taskId: string, sourceType: SourceType) => void;
}

export function ReportPreview({ taskId, isDone, sourceType, onTaskCreated }: Props) {
  const [task, setTask] = useState<Task | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!taskId || !isDone) {
      setTask(null);
      return;
    }

    setLoading(true);
    fetch(`/tasks/${taskId}`)
      .then(r => r.json())
      .then((data: Task) => {
        setTask(data);
      })
      .catch(() =>
        setTask({
          task_id: taskId,
          status: 'failed',
          created_at: '',
          source_type: sourceType ?? undefined,
          error: 'Failed to load result',
        }),
      )
      .finally(() => setLoading(false));
  }, [isDone, sourceType, taskId]);

  const effectiveSourceType = task?.source_type ?? sourceType ?? null;
  const isResearchTask = effectiveSourceType === 'research';

  if (!taskId) {
    return (
      <div className="rounded-xl border border-dashed border-stone-300 bg-white/80 p-8 text-center text-stone-500 shadow-sm">
        <p className="mb-1 text-lg font-display text-stone-600">No task selected</p>
        <p className="text-sm">Start a paper report or research workflow to see the output here.</p>
      </div>
    );
  }

  if (!isDone) {
    return (
      <div className="rounded-xl border border-stone-300 bg-white p-8 text-center text-stone-500 shadow-sm">
        <p className="text-sm animate-pulse">
          {isResearchTask
            ? 'Research brief and search plan will appear when the workflow finishes…'
            : 'Report will appear when the pipeline finishes…'}
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="rounded-xl border border-stone-300 bg-white p-8 shadow-sm">
        <p className="text-sm text-stone-500">Loading…</p>
      </div>
    );
  }

  if (isResearchTask) {
    const researchMarkdown = task?.draft_markdown || task?.full_markdown || task?.result_markdown;
    return (
      <div className="max-h-[600px] overflow-y-auto rounded-xl border border-stone-300 bg-white p-8 shadow-sm">
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-full bg-stone-100 px-2.5 py-1 text-stone-700">
            mode: research
          </span>
          {task?.current_stage && (
            <span className="rounded-full bg-stone-100 px-2.5 py-1 text-stone-700">
              stage: {task.current_stage}
            </span>
          )}
        </div>

        {/* Brief — formatted card */}
        {task?.brief && (() => {
          const b = task.brief as ResearchBrief;
          return (
            <section className="mb-4 rounded-xl border border-stone-200 bg-white p-4">
              <h3 className="mb-2 text-xs font-semibold text-stone-500 uppercase tracking-wider">Research Brief</h3>
              <div className="space-y-1.5">
                <p className="text-sm font-semibold text-stone-800 leading-snug">{b.topic}</p>
                {b.goal && <p className="text-xs text-stone-600 leading-relaxed">{b.goal}</p>}
                {b.sub_questions && b.sub_questions.length > 0 && (
                  <div className="space-y-0.5 mt-2">
                    <span className="text-[10px] font-semibold text-stone-400 uppercase">Sub-questions</span>
                    {b.sub_questions.map((q, i) => (
                      <div key={i} className="flex items-start gap-1.5">
                        <span className="mt-0.5 w-1 h-1 rounded-full bg-stone-400 flex-shrink-0" />
                        <span className="text-xs text-stone-600 leading-snug">{q}</span>
                      </div>
                    ))}
                  </div>
                )}
                {b.focus_dimensions && b.focus_dimensions.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {b.focus_dimensions.map(d => (
                      <span key={d} className="px-1.5 py-0.5 rounded bg-stone-100 text-[10px] text-stone-600 border border-stone-200">
                        {d}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </section>
          );
        })()}

        {/* Search Plan — formatted stats */}
        {task?.search_plan && (() => {
          const p = task.search_plan as SearchPlan;
          const totalQueries = (p.query_groups ?? []).reduce((acc: number, g: SearchQueryGroup) => acc + (g.queries ?? []).length, 0);
          const expectedHits = (p.query_groups ?? []).reduce((acc: number, g: SearchQueryGroup) => acc + (g.expected_hits ?? 0), 0);
          return (
            <section className="mb-4 rounded-xl border border-stone-200 bg-white p-4">
              <h3 className="mb-2 text-xs font-semibold text-stone-500 uppercase tracking-wider">Search Plan</h3>
              <div className="grid grid-cols-3 gap-2 mb-3">
                <div className="text-center p-2 rounded-lg bg-stone-50 border border-stone-200">
                  <div className="text-base font-bold text-stone-800 leading-none">{totalQueries}</div>
                  <div className="text-[9px] text-stone-500 mt-0.5">queries</div>
                </div>
                <div className="text-center p-2 rounded-lg bg-stone-50 border border-stone-200">
                  <div className="text-base font-bold text-stone-800 leading-none">{(p.query_groups ?? []).length}</div>
                  <div className="text-[9px] text-stone-500 mt-0.5">groups</div>
                </div>
                <div className="text-center p-2 rounded-lg bg-stone-50 border border-stone-200">
                  <div className="text-base font-bold text-stone-800 leading-none">{expectedHits}</div>
                  <div className="text-[9px] text-stone-500 mt-0.5">est. hits</div>
                </div>
              </div>
              {p.coverage_strategy && (
                <div className="mb-2">
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200">
                    {p.coverage_strategy}
                  </span>
                </div>
              )}
              {(p.query_groups ?? []).length > 0 && (
                <div className="space-y-1">
                  {(p.query_groups ?? []).slice(0, 4).map((g: SearchQueryGroup) => (
                    <div key={g.group_id} className="flex items-start gap-1.5">
                      <span className="mt-0.5 w-1 h-1 rounded-full bg-stone-400 flex-shrink-0" />
                      <span className="text-[11px] text-stone-600 leading-snug truncate flex-1">
                        {(g.queries ?? [])[0]}
                      </span>
                      {g.expected_hits && (
                        <span className="text-[9px] text-stone-400 flex-shrink-0">~{g.expected_hits}</span>
                      )}
                    </div>
                  ))}
                  {(p.query_groups ?? []).length > 4 && (
                    <span className="text-[9px] text-stone-400 pl-2.5">
                      +{(p.query_groups ?? []).length - 4} more groups
                    </span>
                  )}
                </div>
              )}
            </section>
          );
        })()}

        {task?.rag_result != null && (
          <section className="mb-4 rounded-xl border border-stone-200 bg-white p-4">
            <h3 className="mb-2 text-xs font-semibold text-stone-500 uppercase tracking-wider">RAG Result</h3>
            <div className="text-xs text-stone-600">
              {typeof task.rag_result === 'object' && task.rag_result !== null
                ? `${Object.keys(task.rag_result).length} fields`
                : String(task.rag_result ?? '')}
            </div>
          </section>
        )}

        {researchMarkdown && !researchMarkdown.trim().startsWith('{') && (
          <section className="mb-5 rounded-xl border border-stone-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-stone-800">Research Report</h3>
            <MarkdownRenderer content={researchMarkdown} />
          </section>
        )}

        {task?.brief?.needs_followup && !task?.search_plan && (
          <>
            <div className="mb-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              当前 brief 仍需要人工补充澄清，因此 search plan 暂停在 `clarify` 阶段。
            </div>
            <ResearchFollowupForm brief={task.brief} onTaskCreated={onTaskCreated} />
          </>
        )}

        {task?.error && (
          <div className="mb-5 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
            {task.error}
          </div>
        )}

        {!task?.brief && (
          <div className="rounded-lg border border-stone-200 bg-stone-50 p-4 text-sm text-stone-600">
            {task?.error || task?.result_markdown || 'No research result'}
          </div>
        )}
      </div>
    );
  }

  const activeMarkdown =
    task?.report_mode === 'full'
      ? task?.full_markdown || task?.result_markdown
      : task?.draft_markdown || task?.result_markdown;

  return (
    <div className="max-h-[600px] overflow-y-auto rounded-xl border border-stone-300 bg-white p-8 shadow-sm">
      {task && (
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-full bg-stone-100 px-2.5 py-1 text-stone-700">
            mode: {task.report_mode ?? 'draft'}
          </span>
          <span className="rounded-full bg-stone-100 px-2.5 py-1 text-stone-700">
            paper: {task.paper_type ?? 'regular'}
          </span>
        </div>
      )}
      {task?.paper_type === 'survey' && task?.followup_hints && task.followup_hints.length > 0 && (
        <div className="mb-5 rounded-lg border border-stone-200 bg-stone-50 p-4 text-sm text-stone-700">
          <p className="mb-2 font-medium">建议追问</p>
          <ul className="list-disc pl-5 space-y-1">
            {task.followup_hints.map((hint, idx) => (
              <li key={idx}>{hint}</li>
            ))}
          </ul>
        </div>
      )}
      <MarkdownRenderer content={activeMarkdown || task?.error || 'No result'} />
    </div>
  );
}
