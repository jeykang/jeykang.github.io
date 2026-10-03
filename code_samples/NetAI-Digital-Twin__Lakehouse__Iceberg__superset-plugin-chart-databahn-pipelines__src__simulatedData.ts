/**
 * simulatedData.ts — deterministic placeholder data generator.
 *
 * Produces realistic-looking pipeline run records so the chart renders
 * without any backend connection.  Every value here is fabricated; swap
 * this module out once real Databahn data is available.
 */
import { PipelineRun, PipelineSummary, StepDetail } from './types';

// ── Seed data ────────────────────────────────────────────────────────────────

const PIPELINE_DEFS = [
  { name: 'auto-annotation-lidar', label: 'Auto Annotation — LiDAR', category: 'annotation' },
  { name: 'auto-annotation-camera', label: 'Auto Annotation — Camera', category: 'annotation' },
  { name: 'object-detection-3d', label: '3D Object Detection', category: 'perception' },
  { name: 'lane-detection', label: 'Lane Detection', category: 'perception' },
  { name: 'sensor-fusion', label: 'Sensor Fusion', category: 'fusion' },
  { name: 'map-alignment', label: 'HD Map Alignment', category: 'mapping' },
  { name: 'data-quality-check', label: 'Data Quality Check', category: 'validation' },
  { name: 'frame-extraction', label: 'Frame Extraction', category: 'preprocessing' },
  { name: 'point-cloud-denoise', label: 'Point Cloud Denoising', category: 'preprocessing' },
  { name: 'export-to-kitti', label: 'Export to KITTI Format', category: 'export' },
];

const STEP_NAMES_BY_CATEGORY: Record<string, string[]> = {
  annotation: ['Load frames', 'Run model inference', 'Post-process labels', 'Write annotations', 'Validate output'],
  perception: ['Ingest sensor data', 'Preprocess', 'Model forward pass', 'NMS / filtering', 'Serialize detections'],
  fusion: ['Align timestamps', 'Calibration lookup', 'Fuse modalities', 'Consistency check'],
  mapping: ['Load map tiles', 'GPS alignment', 'ICP registration', 'Publish map diff'],
  validation: ['Schema check', 'Range validation', 'Cross-reference check', 'Generate report'],
  preprocessing: ['Read raw files', 'Apply transforms', 'Write output'],
  export: ['Query source', 'Transform schema', 'Write files', 'Checksum verification'],
};

const STATUSES: PipelineRun['status'][] = ['success', 'success', 'success', 'success', 'failure', 'success', 'cancelled', 'timeout', 'success', 'success'];

const TRIGGER_TYPES: PipelineRun['trigger_type'][] = ['scheduled', 'manual', 'event', 'api', 'scheduled', 'webhook'];

const USERS = ['alice', 'bob', 'ci-bot', 'scheduler', 'charlie'];

// ── Deterministic pseudo-random (so snapshots are stable) ────────────────────

let _seed = 42;
function pseudoRandom(): number {
  _seed = (_seed * 16807 + 0) % 2147483647;
  return (_seed - 1) / 2147483646;
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(pseudoRandom() * arr.length)];
}

// ── Generators ───────────────────────────────────────────────────────────────

function generateSteps(category: string, runStatus: PipelineRun['status']): StepDetail[] {
  const names = STEP_NAMES_BY_CATEGORY[category] ?? ['Step 1', 'Step 2', 'Step 3'];
  let failedStepPicked = false;
  return names.map((name, idx) => {
    let stepStatus: StepDetail['status'] = 'success';
    if (runStatus === 'failure' && !failedStepPicked && pseudoRandom() > 0.5) {
      stepStatus = 'failure';
      failedStepPicked = true;
    } else if (runStatus === 'failure' && idx === names.length - 1 && !failedStepPicked) {
      stepStatus = 'failure';
      failedStepPicked = true;
    }
    if (failedStepPicked && stepStatus !== 'failure') {
      stepStatus = 'skipped';
    }
    const dur = Math.round(pseudoRandom() * 120 + 5);
    return {
      step_name: name,
      step_order: idx + 1,
      status: stepStatus,
      duration_seconds: stepStatus !== 'skipped' ? dur : 0,
      step_metrics: stepStatus === 'success' ? { items_processed: Math.round(pseudoRandom() * 5000) } : undefined,
    };
  });
}

export function generateSimulatedRuns(count: number = 80): PipelineRun[] {
  _seed = 42; // reset for determinism
  const now = Date.now();
  const runs: PipelineRun[] = [];

  for (let i = 0; i < count; i++) {
    const def = pick(PIPELINE_DEFS);
    const status = pick(STATUSES);
    const triggeredMs = now - Math.round(pseudoRandom() * 7 * 24 * 3600 * 1000);
    const durationSec = Math.round(pseudoRandom() * 600 + 10);
    const startedMs = triggeredMs + Math.round(pseudoRandom() * 30000);
    const finishedMs = startedMs + durationSec * 1000;

    runs.push({
      run_id: `run-${String(i).padStart(4, '0')}`,
      pipeline_name: def.name,
      pipeline_label: def.label,
      pipeline_category: def.category,
      triggered_at: new Date(triggeredMs).toISOString(),
      started_at: new Date(startedMs).toISOString(),
      finished_at: status !== 'running' ? new Date(finishedMs).toISOString() : undefined,
      duration_seconds: status !== 'running' ? durationSec : undefined,
      status,
      error_message: status === 'failure' ? 'Simulated error — replace with real data' : undefined,
      error_type: status === 'failure' ? pick(['OOM', 'timeout', 'user-code', 'infra']) : undefined,
      retry_count: status === 'failure' ? Math.floor(pseudoRandom() * 3) : 0,
      triggered_by: pick(USERS),
      trigger_type: pick(TRIGGER_TYPES),
      pipeline_version: `v1.${Math.floor(pseudoRandom() * 10)}`,
      input_datasets: [`dataset-${Math.floor(pseudoRandom() * 20)}`],
      output_datasets: status === 'success' ? [`output-${Math.floor(pseudoRandom() * 20)}`] : undefined,
      compute_environment: pick(['gpu-a100', 'gpu-v100', 'cpu-pool-1', 'cpu-pool-2']),
      resource_usage: status === 'success' ? {
        cpu_seconds: Math.round(pseudoRandom() * 3600),
        peak_memory_mb: Math.round(pseudoRandom() * 16384),
        gpu_hours: Math.round(pseudoRandom() * 4 * 100) / 100,
      } : undefined,
      steps: generateSteps(def.category, status),
    });
  }

  // Sort newest-first
  runs.sort((a, b) => new Date(b.triggered_at).getTime() - new Date(a.triggered_at).getTime());
  return runs;
}

// ── Derive summaries from run records ────────────────────────────────────────

export function deriveSummaries(runs: PipelineRun[]): PipelineSummary[] {
  const grouped = new Map<string, PipelineRun[]>();
  for (const r of runs) {
    const arr = grouped.get(r.pipeline_name) ?? [];
    arr.push(r);
    grouped.set(r.pipeline_name, arr);
  }

  const summaries: PipelineSummary[] = [];
  for (const [name, pipelineRuns] of grouped.entries()) {
    const terminal = pipelineRuns.filter(r => !['running', 'queued'].includes(r.status));
    const successful = terminal.filter(r => r.status === 'success');
    const failed = terminal.filter(r => r.status === 'failure');
    const cancelled = terminal.filter(r => r.status === 'cancelled');
    const timeout = terminal.filter(r => r.status === 'timeout');
    const running = pipelineRuns.filter(r => r.status === 'running');
    const queued = pipelineRuns.filter(r => r.status === 'queued');

    const durations = terminal
      .map(r => r.duration_seconds)
      .filter((d): d is number => d != null)
      .sort((a, b) => a - b);

    const avgDur = durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;

    const pct = (idx: number) => durations.length > 0 ? durations[Math.min(Math.floor(durations.length * idx), durations.length - 1)] : 0;

    const last = pipelineRuns[0]; // already sorted newest-first

    summaries.push({
      pipeline_name: name,
      pipeline_label: pipelineRuns[0]?.pipeline_label,
      pipeline_category: pipelineRuns[0]?.pipeline_category,
      total_runs: pipelineRuns.length,
      successful_runs: successful.length,
      failed_runs: failed.length,
      cancelled_runs: cancelled.length,
      timeout_runs: timeout.length,
      running_now: running.length,
      queued_now: queued.length,
      success_rate: terminal.length > 0 ? successful.length / terminal.length : 0,
      avg_duration_seconds: Math.round(avgDur),
      p50_duration_seconds: pct(0.5),
      p90_duration_seconds: pct(0.9),
      p99_duration_seconds: pct(0.99),
      last_run_at: last?.triggered_at,
      last_run_status: last?.status,
    });
  }

  summaries.sort((a, b) => b.total_runs - a.total_runs);
  return summaries;
}
