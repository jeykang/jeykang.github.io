/**
 * DatabannPipelinesChart.tsx — main React component for the plugin.
 *
 * Renders one of five display modes depending on the user's control-panel
 * selection.  All rendering uses basic HTML/CSS (no external charting lib)
 * so the template has zero extra dependencies.  Swap in ECharts, D3, etc.
 * once the real integration is underway.
 */
import React, { useMemo } from 'react';
import { styled } from '@superset-ui/core';
import {
  DatabannPipelinesChartProps,
  PipelineRun,
  PipelineSummary,
} from './types';

// ── Styled root container ────────────────────────────────────────────────────

const Styles = styled.div<{ width: number; height: number }>`
  width: ${({ width }) => width}px;
  height: ${({ height }) => height}px;
  overflow: auto;
  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 13px;
  color: #333;
  padding: 12px;
  box-sizing: border-box;

  h3 {
    margin: 0 0 12px 0;
    font-size: 15px;
    font-weight: 600;
    color: #1a1a2e;
  }

  .db-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .db-badge--success  { background: #d4edda; color: #155724; }
  .db-badge--failure  { background: #f8d7da; color: #721c24; }
  .db-badge--running  { background: #cce5ff; color: #004085; }
  .db-badge--queued   { background: #e2e3e5; color: #383d41; }
  .db-badge--cancelled{ background: #fff3cd; color: #856404; }
  .db-badge--timeout  { background: #f5c6cb; color: #721c24; }
  .db-badge--skipped  { background: #e2e3e5; color: #6c757d; }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 16px;
  }
  th, td {
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #e9ecef;
    white-space: nowrap;
  }
  th {
    background: #f1f3f5;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #495057;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  tr:hover td { background: #f8f9fa; }

  .db-bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 4px 0;
  }
  .db-bar-label { min-width: 180px; font-size: 12px; }
  .db-bar-track {
    flex: 1;
    height: 18px;
    background: #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
  }
  .db-bar-segment {
    height: 100%;
    transition: width 0.3s ease;
  }
  .db-bar-value { min-width: 50px; font-size: 12px; text-align: right; }

  .db-pie-container {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
    justify-content: center;
  }
  .db-pie-legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .db-pie-legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
  }
  .db-pie-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }

  .db-hist-bar {
    display: inline-block;
    background: #4263eb;
    border-radius: 3px 3px 0 0;
    margin: 0 2px;
    min-width: 20px;
    vertical-align: bottom;
  }
  .db-hist-container {
    display: flex;
    align-items: flex-end;
    height: 140px;
    border-bottom: 2px solid #adb5bd;
    padding: 0 4px;
    margin-bottom: 4px;
  }
  .db-hist-labels {
    display: flex;
    padding: 0 4px;
    font-size: 10px;
    color: #868e96;
  }
  .db-hist-labels span {
    min-width: 24px;
    text-align: center;
    margin: 0 2px;
  }

  .db-gantt-row {
    display: flex;
    align-items: center;
    margin: 3px 0;
  }
  .db-gantt-label { min-width: 180px; font-size: 12px; }
  .db-gantt-track {
    flex: 1;
    height: 20px;
    background: #f1f3f5;
    border-radius: 3px;
    position: relative;
  }
  .db-gantt-bar {
    position: absolute;
    height: 100%;
    border-radius: 3px;
    opacity: 0.85;
  }

  .db-placeholder-msg {
    color: #868e96;
    font-style: italic;
    text-align: center;
    margin-top: 40px;
  }
`;

// ── Status colour map ────────────────────────────────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  success: '#28a745',
  failure: '#dc3545',
  running: '#007bff',
  queued: '#6c757d',
  cancelled: '#ffc107',
  timeout: '#e83e8c',
  skipped: '#adb5bd',
};

// ── Sub-components (one per display mode) ────────────────────────────────────

/** 1. Summary table — one row per pipeline with aggregated stats */
function SummaryTable({ summaries }: { summaries: PipelineSummary[] }) {
  if (summaries.length === 0) {
    return <p className="db-placeholder-msg">No pipeline data available. Connect a Databahn data source to populate this chart.</p>;
  }
  return (
    <>
      <h3>Pipeline Summary</h3>
      <table>
        <thead>
          <tr>
            <th>Pipeline</th>
            <th>Category</th>
            <th>Total</th>
            <th>✓</th>
            <th>✗</th>
            <th>Rate</th>
            <th>Avg Dur.</th>
            <th>p90</th>
            <th>Running</th>
            <th>Last Status</th>
          </tr>
        </thead>
        <tbody>
          {summaries.map(s => (
            <tr key={s.pipeline_name}>
              <td title={s.pipeline_name}>{s.pipeline_label ?? s.pipeline_name}</td>
              <td>{s.pipeline_category ?? '—'}</td>
              <td>{s.total_runs}</td>
              <td>{s.successful_runs}</td>
              <td>{s.failed_runs}</td>
              <td>{(s.success_rate * 100).toFixed(1)}%</td>
              <td>{formatDuration(s.avg_duration_seconds)}</td>
              <td>{s.p90_duration_seconds != null ? formatDuration(s.p90_duration_seconds) : '—'}</td>
              <td>{s.running_now}</td>
              <td>
                <span className={`db-badge db-badge--${s.last_run_status ?? 'queued'}`}>
                  {s.last_run_status ?? '—'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** 2. Stacked horizontal bars — shows status breakdown per pipeline */
function StatusBreakdown({ summaries }: { summaries: PipelineSummary[] }) {
  const statuses = ['success', 'failure', 'cancelled', 'timeout'] as const;

  return (
    <>
      <h3>Status Breakdown</h3>
      <div className="db-pie-container" style={{ justifyContent: 'flex-start', flexDirection: 'column', width: '100%' }}>
        {summaries.map(s => {
          const total = s.total_runs || 1;
          return (
            <div key={s.pipeline_name} className="db-bar-row">
              <span className="db-bar-label">{s.pipeline_label ?? s.pipeline_name}</span>
              <div className="db-bar-track">
                {statuses.map(st => {
                  const count = st === 'success' ? s.successful_runs
                    : st === 'failure' ? s.failed_runs
                    : st === 'cancelled' ? s.cancelled_runs
                    : s.timeout_runs;
                  const pct = (count / total) * 100;
                  return pct > 0 ? (
                    <div
                      key={st}
                      className="db-bar-segment"
                      style={{ width: `${pct}%`, background: STATUS_COLORS[st] }}
                      title={`${st}: ${count} (${pct.toFixed(1)}%)`}
                    />
                  ) : null;
                })}
              </div>
              <span className="db-bar-value">{s.total_runs} runs</span>
            </div>
          );
        })}
      </div>
      <div className="db-pie-legend" style={{ flexDirection: 'row', gap: 16, marginTop: 12 }}>
        {statuses.map(st => (
          <div key={st} className="db-pie-legend-item">
            <div className="db-pie-swatch" style={{ background: STATUS_COLORS[st] }} />
            <span>{st}</span>
          </div>
        ))}
      </div>
    </>
  );
}

/** 3. Run timeline — simple table listing of runs sorted by time */
function RunTimeline({ runs }: { runs: PipelineRun[] }) {
  if (runs.length === 0) {
    return <p className="db-placeholder-msg">No runs in the selected window.</p>;
  }
  return (
    <>
      <h3>Run Timeline</h3>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Pipeline</th>
            <th>Triggered</th>
            <th>Duration</th>
            <th>Status</th>
            <th>Trigger</th>
            <th>User</th>
            <th>Env</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(r => (
            <tr key={r.run_id}>
              <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.run_id}</td>
              <td>{r.pipeline_label ?? r.pipeline_name}</td>
              <td>{new Date(r.triggered_at).toLocaleString()}</td>
              <td>{r.duration_seconds != null ? formatDuration(r.duration_seconds) : '—'}</td>
              <td>
                <span className={`db-badge db-badge--${r.status}`}>{r.status}</span>
              </td>
              <td>{r.trigger_type ?? '—'}</td>
              <td>{r.triggered_by ?? '—'}</td>
              <td>{r.compute_environment ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** 4. Duration histogram */
function DurationHistogram({ runs }: { runs: PipelineRun[] }) {
  const durations = runs
    .map(r => r.duration_seconds)
    .filter((d): d is number => d != null);

  if (durations.length === 0) {
    return <p className="db-placeholder-msg">No duration data available.</p>;
  }

  const maxDur = Math.max(...durations);
  const bucketCount = 12;
  const bucketSize = Math.ceil(maxDur / bucketCount) || 1;
  const buckets = new Array(bucketCount).fill(0);
  for (const d of durations) {
    const idx = Math.min(Math.floor(d / bucketSize), bucketCount - 1);
    buckets[idx]++;
  }
  const maxBucket = Math.max(...buckets, 1);

  return (
    <>
      <h3>Duration Distribution</h3>
      <div className="db-hist-container">
        {buckets.map((count, idx) => (
          <div
            key={idx}
            className="db-hist-bar"
            style={{
              height: `${(count / maxBucket) * 100}%`,
              flex: 1,
            }}
            title={`${formatDuration(idx * bucketSize)}–${formatDuration((idx + 1) * bucketSize)}: ${count} runs`}
          />
        ))}
      </div>
      <div className="db-hist-labels">
        {buckets.map((_, idx) => (
          <span key={idx} style={{ flex: 1 }}>
            {formatDuration(idx * bucketSize)}
          </span>
        ))}
      </div>
    </>
  );
}

/** 5. Step Gantt chart — shows per-step duration breakdown for recent runs */
function StepGantt({ runs }: { runs: PipelineRun[] }) {
  // Pick the most recent run that has steps
  const run = runs.find(r => r.steps && r.steps.length > 0);
  if (!run || !run.steps) {
    return <p className="db-placeholder-msg">No step-level data available. Enable "Show Step Details" and ensure runs contain step breakdowns.</p>;
  }

  const totalDur = run.steps.reduce((acc, s) => acc + (s.duration_seconds ?? 0), 0) || 1;
  let cumulative = 0;

  return (
    <>
      <h3>Step Breakdown — {run.pipeline_label ?? run.pipeline_name} ({run.run_id})</h3>
      {run.steps.map(step => {
        const pct = ((step.duration_seconds ?? 0) / totalDur) * 100;
        const left = (cumulative / totalDur) * 100;
        cumulative += step.duration_seconds ?? 0;
        return (
          <div key={step.step_name} className="db-gantt-row">
            <span className="db-gantt-label">{step.step_order}. {step.step_name}</span>
            <div className="db-gantt-track">
              <div
                className="db-gantt-bar"
                style={{
                  left: `${left}%`,
                  width: `${pct}%`,
                  background: STATUS_COLORS[step.status] ?? '#6c757d',
                }}
                title={`${step.step_name}: ${formatDuration(step.duration_seconds ?? 0)} (${step.status})`}
              />
            </div>
          </div>
        );
      })}
    </>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

// ── Main component ──────────────────────────────────────────────────────────

export default function DatabannPipelinesChart(props: DatabannPipelinesChartProps) {
  const { width, height, formData, runs, summaries } = props;
  const mode = formData.display_mode ?? 'summary_table';

  const content = useMemo(() => {
    switch (mode) {
      case 'summary_table':
        return <SummaryTable summaries={summaries} />;
      case 'status_breakdown':
        return <StatusBreakdown summaries={summaries} />;
      case 'timeline':
        return <RunTimeline runs={runs} />;
      case 'duration_histogram':
        return <DurationHistogram runs={runs} />;
      case 'step_gantt':
        return <StepGantt runs={runs} />;
      default:
        return <SummaryTable summaries={summaries} />;
    }
  }, [mode, runs, summaries]);

  return (
    <Styles width={width} height={height}>
      {content}
    </Styles>
  );
}
