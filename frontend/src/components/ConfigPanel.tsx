/**
 * ConfigPanel — Execution mode & feature toggles.
 *
 * Design: "Precision Controls"
 * - Segmented mode selector with spring press feedback
 * - Pill-style feature toggles with micro-bounce
 * - Save button with state transitions
 */

import { useState, useEffect } from 'react';
import type { Phase4Config, ExecutionMode, SupervisorMode } from '../types/phase34';

interface Props {
  onConfigChange?: (config: Phase4Config) => void;
}

const MODES: { id: ExecutionMode; label: string; desc: string; accent: string }[] = [
  {
    id: 'legacy',
    label: 'Legacy',
    desc: 'Phase 1-3 stable path',
    accent: '#78716c',
  },
  {
    id: 'hybrid',
    label: 'Hybrid',
    desc: 'New agents, auto fallback',
    accent: '#92400e',
  },
  {
    id: 'v2',
    label: 'v2',
    desc: 'Full Phase 4 pipeline',
    accent: '#1e3a5f',
  },
];

const SUPERVISOR_MODES: { id: SupervisorMode; label: string; desc: string; accent: string }[] = [
  {
    id: 'graph',
    label: 'Graph',
    desc: 'Deterministic LangGraph routing',
    accent: '#1e3a5f',
  },
  {
    id: 'llm_handoff',
    label: 'LLM Handoff',
    desc: 'Official LangGraph supervisor',
    accent: '#065f46',
  },
];

const FEATURES = [
  {
    key: 'enable_mcp' as const,
    label: 'MCP',
    desc: 'Model Context Protocol tools',
    color: '#1e3a5f',
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
        <path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
      </svg>
    ),
  },
  {
    key: 'enable_skills' as const,
    label: 'Skills',
    desc: 'Agent Skills system',
    color: '#065f46',
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
        <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
      </svg>
    ),
  },
  {
    key: 'enable_replan' as const,
    label: 'Re-plan',
    desc: 'Auto re-plan on review',
    color: '#6d28d9',
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
    ),
  },
  {
    key: 'auto_fill' as const,
    label: 'Auto-fill',
    desc: 'LLM auto-completes ambiguous fields',
    color: '#0e7490',
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707"/>
        <circle cx="12" cy="12" r="4"/>
      </svg>
    ),
  },
];

export function ConfigPanel({ onConfigChange }: Props) {
  const [config, setConfig] = useState<Phase4Config | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/v1/config/phase4')
      .then(r => r.json())
      .then(d => setConfig(d.config))
      .catch(() => setError('Failed to load config'));
  }, []);

  if (error) return <p className="text-xs text-rose-600">{error}</p>;
  if (!config) return (
    <div className="flex flex-col gap-3">
      <div className="h-20 rounded-xl bg-stone-100 animate-pulse" />
      <div className="h-12 rounded-xl bg-stone-100 animate-pulse" />
    </div>
  );

  const update = (patch: Partial<Phase4Config>) => {
    const next = { ...config, ...patch };
    setConfig(next);
    onConfigChange?.(next);
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch('/api/v1/config/phase4', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });
      if (res.ok) {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      } else {
        setError('Save failed');
      }
    } catch {
      setError('Network error');
    } finally {
      setSaving(false);
    }
  };

  const activeMode = MODES.find(m => m.id === config.execution_mode)!;

  return (
    <div className="flex flex-col gap-5">
      {/* Mode selector */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-0.5 h-4 rounded-full" style={{ backgroundColor: activeMode.accent }} />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">
            Execution Mode
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {MODES.map(mode => {
            const active = config.execution_mode === mode.id;
            return (
              <button
                key={mode.id}
                onClick={() => update({ execution_mode: mode.id })}
                className={`
                  relative flex flex-col items-center gap-0.5 px-3 py-2.5 rounded-xl border
                  transition-all duration-150 cursor-pointer text-center overflow-hidden
                  ${active
                    ? 'border-2 shadow-sm'
                    : 'border border-stone-200 hover:border-stone-300 hover:shadow-sm'
                  }
                `}
                style={active ? {
                  borderColor: mode.accent,
                  backgroundColor: `${mode.accent}0a`,
                  boxShadow: `0 0 0 1px ${mode.accent}20`,
                } : {}}
              >
                {active && (
                  <div
                    className="absolute inset-0 opacity-5"
                    style={{ background: `radial-gradient(ellipse at top, ${mode.accent}, transparent 70%)` }}
                  />
                )}
                <span
                  className="text-[11px] font-bold relative z-10"
                  style={active ? { color: mode.accent } : { color: '#78716c' }}
                >
                  {mode.label}
                </span>
                <span className="text-[9px] text-stone-400 leading-tight relative z-10">
                  {mode.desc}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Supervisor mode */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-0.5 h-4 rounded-full" style={{ backgroundColor: activeMode.accent }} />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">
            Supervisor
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {SUPERVISOR_MODES.map(mode => {
            const active = config.supervisor_mode === mode.id;
            return (
              <button
                key={mode.id}
                onClick={() => update({ supervisor_mode: mode.id })}
                className={`
                  relative flex flex-col items-start gap-0.5 px-3 py-2.5 rounded-xl border
                  transition-all duration-150 cursor-pointer text-left overflow-hidden
                  ${active
                    ? 'border-2 shadow-sm'
                    : 'border border-stone-200 hover:border-stone-300 hover:shadow-sm'
                  }
                `}
                style={active ? {
                  borderColor: mode.accent,
                  backgroundColor: `${mode.accent}0a`,
                  boxShadow: `0 0 0 1px ${mode.accent}20`,
                } : {}}
              >
                {active && (
                  <div
                    className="absolute inset-0 opacity-5"
                    style={{ background: `radial-gradient(ellipse at top left, ${mode.accent}, transparent 70%)` }}
                  />
                )}
                <span
                  className="text-[11px] font-bold relative z-10"
                  style={active ? { color: mode.accent } : { color: '#44403c' }}
                >
                  {mode.label}
                </span>
                <span className="text-[9px] text-stone-400 leading-tight relative z-10">
                  {mode.desc}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Feature toggles */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-0.5 h-4 rounded-full" style={{ backgroundColor: activeMode.accent }} />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">
            Features
          </span>
        </div>
        <div className="flex flex-col gap-2">
          {FEATURES.map(feat => {
            const active = config[feat.key] as boolean;
            return (
              <button
                key={feat.key}
                onClick={() => update({ [feat.key]: !active })}
                title={feat.desc}
                className={`
                  flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border
                  transition-all duration-150 cursor-pointer group
                  ${active
                    ? 'border-2 shadow-sm'
                    : 'border border-stone-200 hover:border-stone-300 hover:shadow-sm'
                  }
                `}
                style={active ? {
                  borderColor: `${feat.color}40`,
                  backgroundColor: `${feat.color}08`,
                } : {}}
              >
                <div className="flex items-center gap-2.5">
                  <span style={{ color: feat.color }}>{feat.icon}</span>
                  <div className="text-left">
                    <p className="text-[11px] font-semibold" style={active ? { color: feat.color } : { color: '#44403c' }}>
                      {feat.label}
                    </p>
                    <p className="text-[9px] text-stone-400 leading-tight">{feat.desc}</p>
                  </div>
                </div>
                <div
                  className={`
                    w-8 h-4 rounded-full border transition-all duration-200 relative flex-shrink-0
                    ${active ? 'border-transparent' : 'border-stone-300 bg-stone-100'}
                  `}
                  style={active ? { backgroundColor: feat.color } : {}}
                >
                  <div
                    className={`
                      absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm
                      transition-transform duration-200
                      ${active ? 'translate-x-4' : 'translate-x-0.5'}
                    `}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Save */}
      {error && (
        <div className="px-3 py-2 rounded-lg bg-rose-50 border border-rose-200 text-[10px] text-rose-700">
          {error}
        </div>
      )}
      <button
        onClick={save}
        disabled={saving}
        className={`
          relative flex items-center justify-center gap-2 py-2.5 rounded-xl border
          text-[11px] font-semibold transition-all duration-200 cursor-pointer overflow-hidden
          ${saved
            ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
            : saving
              ? 'border-stone-200 bg-stone-50 text-stone-400'
              : 'border-stone-300 text-stone-700 hover:border-stone-400 hover:shadow-sm active:scale-95'
          }
        `}
      >
        {saving && (
          <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 2v4m0 12v4m-8-10H2m20 0h-2M6.34 6.34L4.93 4.93m14.14 14.14l-1.41-1.41M6.34 17.66l-1.41 1.41"/>
          </svg>
        )}
        {saved ? '✓ Saved' : saving ? 'Saving…' : 'Save Config'}
      </button>
    </div>
  );
}
