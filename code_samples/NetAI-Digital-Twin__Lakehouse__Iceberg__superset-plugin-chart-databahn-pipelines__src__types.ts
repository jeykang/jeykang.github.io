/**
 * Type definitions for the Databahn Pipelines plugin.
 *
 * These types are intentionally broad so that no data shape is prematurely
 * excluded.  Once the real Databahn database / API schema is known, narrow
 * them down and remove anything superfluous.
 */

// ---------------------------------------------------------------------------
// 1. Individual pipeline run record
//    Represents a single execution of a workflow / pipeline.
// ---------------------------------------------------------------------------
export interface PipelineRun {
  /** Unique identifier for this execution */
  run_id: string;
  /** Name / slug of the pipeline definition */
  pipeline_name: string;
  /** Human-readable pipeline label (may differ from slug) */
  pipeline_label?: string;
  /** Pipeline category / group (e.g. "annotation", "object-detection") */
  pipeline_category?: string;
  /** ISO-8601 timestamp — when the run was enqueued / triggered */
  triggered_at: string;
  /** ISO-8601 timestamp — when execution actually started */
  started_at?: string;
  /** ISO-8601 timestamp — when execution finished */
  finished_at?: string;
  /** Wall-clock duration in seconds (derived or stored) */
  duration_seconds?: number;
  /** Terminal status of the run */
  status: 'success' | 'failure' | 'running' | 'queued' | 'cancelled' | 'timeout' | 'skipped';
  /** Free-form error message / traceback (only for failed runs) */
  error_message?: string;
  /** Error classification (e.g. "OOM", "timeout", "user-code") */
  error_type?: string;
  /** Number of automatic retry attempts consumed */
  retry_count?: number;
  /** User or service account that triggered the run */
  triggered_by?: string;
  /** Trigger mechanism */
  trigger_type?: 'manual' | 'scheduled' | 'event' | 'api' | 'webhook';
  /** Git / version reference for the pipeline definition used */
  pipeline_version?: string;
  /** Arbitrary key-value metadata attached to the run */
  run_metadata?: Record<string, unknown>;
  /** Input dataset identifier(s) */
  input_datasets?: string[];
  /** Output dataset identifier(s) */
  output_datasets?: string[];
  /** Compute environment label (e.g. "gpu-a100", "cpu-pool-1") */
  compute_environment?: string;
  /** Resource usage metrics (CPU-seconds, peak-memory-MB, GPU-hours, …) */
  resource_usage?: Record<string, number>;
  /** Per-step breakdown (see StepDetail) */
  steps?: StepDetail[];
}

// ---------------------------------------------------------------------------
// 2. Per-step breakdown within a single pipeline run
// ---------------------------------------------------------------------------
export interface StepDetail {
  step_name: string;
  step_order: number;
  status: 'success' | 'failure' | 'running' | 'skipped';
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  error_message?: string;
  /** Metrics emitted by this step (e.g. {"objects_detected": 1423}) */
  step_metrics?: Record<string, number>;
}

// ---------------------------------------------------------------------------
// 3. Aggregated pipeline summary (pre-computed or derived)
// ---------------------------------------------------------------------------
export interface PipelineSummary {
  pipeline_name: string;
  pipeline_label?: string;
  pipeline_category?: string;
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  cancelled_runs: number;
  timeout_runs: number;
  running_now: number;
  queued_now: number;
  /** Computed success rate (0–1) */
  success_rate: number;
  /** Average wall-clock duration in seconds across terminal runs */
  avg_duration_seconds: number;
  /** p50, p90, p99 duration */
  p50_duration_seconds?: number;
  p90_duration_seconds?: number;
  p99_duration_seconds?: number;
  /** Timestamp of the most recent completed run */
  last_run_at?: string;
  last_run_status?: string;
  /** Throughput — runs completed per hour over the window */
  throughput_per_hour?: number;
}

// ---------------------------------------------------------------------------
// 4. Chart form data — mirrors what the user selects in the control panel
// ---------------------------------------------------------------------------
export interface DatabannPipelinesFormData {
  /** Which display mode the user picked */
  display_mode: 'summary_table' | 'timeline' | 'status_breakdown' | 'duration_histogram' | 'step_gantt';
  /** Filter to a specific pipeline name (empty = all) */
  pipeline_filter?: string;
  /** Filter to a specific status */
  status_filter?: string;
  /** Lookback window in hours */
  lookback_hours?: number;
  /** Whether to show per-step drill-down */
  show_steps?: boolean;
  /** Number of rows to display (for table modes) */
  row_limit?: number;
  /** Color scheme override */
  color_scheme?: string;
}

// ---------------------------------------------------------------------------
// 5. Props passed to the React component after transformProps
// ---------------------------------------------------------------------------
export interface DatabannPipelinesChartProps {
  width: number;
  height: number;
  formData: DatabannPipelinesFormData;
  /** Flat run-level rows from the query result */
  runs: PipelineRun[];
  /** Pre-aggregated summaries (may come from a different query or be derived) */
  summaries: PipelineSummary[];
}
