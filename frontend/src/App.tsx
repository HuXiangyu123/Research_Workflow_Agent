import { useState } from 'react';
import { TaskSubmitForm } from './components/TaskSubmitForm';
import { GraphView } from './components/GraphView';
import { ProgressBar } from './components/ProgressBar';
import { ReportPreview } from './components/ReportPreview';
import { TaskHistory } from './components/TaskHistory';
import { ChatPanel } from './components/ChatPanel';
import { ThinkingPanel } from './components/ThinkingPanel';
import { WorkspaceInspectorPanel } from './components/WorkspaceInspectorPanel';
import { useTaskSSE } from './hooks/useTaskSSE';
import type { SourceType, WorkflowMode } from './types/task';
import './index.css';

function App() {
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [selectedTaskSourceType, setSelectedTaskSourceType] = useState<SourceType | null>(null);
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>('report');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const sse = useTaskSSE(activeTaskId);
  const formSourceType: SourceType = workflowMode === 'research' ? 'research' : 'arxiv';
  const taskSourceType = sse.sourceType ?? selectedTaskSourceType;
  const effectiveSourceType = activeTaskId ? taskSourceType ?? formSourceType : formSourceType;
  const isResearchMode = effectiveSourceType === 'research';

  const handleTaskCreated = (taskId: string, sourceType: SourceType) => {
    setActiveTaskId(taskId);
    setSelectedTaskSourceType(sourceType);
    setWorkflowMode(sourceType === 'research' ? 'research' : 'report');
    setRefreshTrigger(n => n + 1);
  };

  const handleTaskSelected = (taskId: string, sourceType?: SourceType) => {
    setActiveTaskId(taskId);
    setSelectedTaskSourceType(sourceType ?? null);
    if (sourceType) {
      setWorkflowMode(sourceType === 'research' ? 'research' : 'report');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#f0ebe3] text-stone-900">
      <header className="border-b border-stone-300/80 bg-white/90 backdrop-blur-sm px-6 py-5 shadow-sm">
        <h1 className="font-display text-2xl font-semibold tracking-tight text-[#1e3a5f]">
          Literature Report Agent
        </h1>
        <p className="text-sm text-stone-600 mt-1.5 max-w-2xl leading-relaxed">
          {isResearchMode
            ? 'Research workflow · Clarify brief → Search plan · Real-time status'
            : '11-node StateGraph · Citation verification · Real-time pipeline'}
        </p>
      </header>

      <div className="px-6 py-4 border-b border-stone-300/60 bg-white/60">
        <TaskSubmitForm
          onTaskCreated={handleTaskCreated}
          workflowMode={workflowMode}
          onWorkflowModeChange={setWorkflowMode}
          workspaceId={sse.workspaceId}
        />
      </div>

      {activeTaskId && (
        <div className="px-6 py-3 bg-[#e8e2d8]/50 border-b border-stone-300/50">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full border border-stone-300 bg-white/80 px-2.5 py-1 font-medium text-stone-700">
              {isResearchMode ? 'Research Mode' : 'Report Mode'}
            </span>
            {sse.currentStage && (
              <span className="rounded-full border border-stone-300 bg-white/80 px-2.5 py-1 text-stone-600">
                stage: {sse.currentStage}
              </span>
            )}
            {sse.taskStatus && (
              <span className="rounded-full border border-stone-300 bg-white/80 px-2.5 py-1 text-stone-600">
                status: {sse.taskStatus}
              </span>
            )}
          </div>
          <ProgressBar
            nodeStatuses={sse.nodeStatuses}
            sourceType={effectiveSourceType}
            currentStage={sse.currentStage}
          />
        </div>
      )}

      <div className="flex-1 flex min-h-0 px-6 py-5 gap-5">
        <div className="w-72 flex-shrink-0">
          <div className="h-[660px] rounded-xl border border-stone-300 bg-white shadow-sm overflow-hidden">
            <GraphView nodeStatuses={sse.nodeStatuses} sourceType={effectiveSourceType} />
          </div>
          <div className="mt-4">
            <TaskHistory onSelect={handleTaskSelected} refreshTrigger={refreshTrigger} />
          </div>
        </div>

        <div className="flex-1 min-w-0">
          {activeTaskId && (
            <ThinkingPanel
              thinkingEntries={sse.thinkingEntries}
              totalDurationMs={sse.totalDurationMs}
              isDone={sse.isDone}
              isRunning={sse.taskStatus === 'running'}
            />
          )}
          <ReportPreview
            taskId={activeTaskId}
            isDone={sse.isDone}
            sourceType={effectiveSourceType}
            workspaceId={sse.workspaceId}
            liveMarkdown={sse.latestReportMarkdown}
            liveArtifactName={sse.latestReportArtifact}
            currentStage={sse.currentStage}
            taskStatus={sse.taskStatus}
            onTaskCreated={handleTaskCreated}
          />
          <ChatPanel taskId={activeTaskId} isDone={sse.isDone} />
        </div>

        <div className="w-80 flex-shrink-0">
          <div className="h-[660px]">
            <WorkspaceInspectorPanel
              taskId={activeTaskId}
              workspaceId={sse.workspaceId}
              isRunning={sse.taskStatus === 'running'}
              events={sse.events}
              highlightArtifact={sse.latestReportArtifact}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
