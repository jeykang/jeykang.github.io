/**
 * transformProps — receives the raw Superset ChartProps (which wraps the
 * backend query result) and returns the props that the React component
 * actually consumes.
 *
 * While the real data source is unknown, this function generates *simulated*
 * placeholder data so the chart renders something useful during development.
 * Replace the body with a real mapping once Databahn is connected.
 */
import { ChartProps } from '@superset-ui/core';
import {
  DatabannPipelinesChartProps,
  DatabannPipelinesFormData,
  PipelineRun,
  PipelineSummary,
} from '../types';
import { generateSimulatedRuns, deriveSummaries } from '../simulatedData';

export default function transformProps(chartProps: ChartProps): DatabannPipelinesChartProps {
  const { width, height, formData, queriesData } = chartProps;
  const fd = formData as unknown as DatabannPipelinesFormData;

  // ------------------------------------------------------------------
  // If real query data is present, use it.  Otherwise fall back to the
  // built-in simulated data so the plugin always renders.
  // ------------------------------------------------------------------
  let runs: PipelineRun[];

  const rawData: any[] | undefined = queriesData?.[0]?.data as any[];
  if (rawData && rawData.length > 0) {
    // Map raw SQL/API rows → PipelineRun objects.
    // This mapping will need to be adjusted to match the actual Databahn
    // column names once they are known.
    runs = rawData.map((row: any) => ({
      run_id: row.run_id ?? row.id ?? String(Math.random()),
      pipeline_name: row.pipeline_name ?? row.name ?? 'unknown',
      pipeline_label: row.pipeline_label,
      pipeline_category: row.pipeline_category ?? row.category,
      triggered_at: row.triggered_at ?? row.created_at ?? new Date().toISOString(),
      started_at: row.started_at,
      finished_at: row.finished_at ?? row.completed_at,
      duration_seconds: row.duration_seconds ?? row.duration,
      status: row.status ?? 'success',
      error_message: row.error_message ?? row.error,
      error_type: row.error_type,
      retry_count: row.retry_count ?? row.retries,
      triggered_by: row.triggered_by ?? row.user,
      trigger_type: row.trigger_type,
      pipeline_version: row.pipeline_version ?? row.version,
      input_datasets: row.input_datasets,
      output_datasets: row.output_datasets,
      compute_environment: row.compute_environment,
      resource_usage: row.resource_usage,
      steps: row.steps,
    }));
  } else {
    // ── Simulated placeholder data ────────────────────────────────────
    runs = generateSimulatedRuns();
  }

  // Apply client-side filters from the control panel
  if (fd.pipeline_filter) {
    const f = fd.pipeline_filter.toLowerCase();
    runs = runs.filter(r => r.pipeline_name.toLowerCase().includes(f));
  }
  if (fd.status_filter) {
    runs = runs.filter(r => r.status === fd.status_filter);
  }
  if (fd.row_limit) {
    runs = runs.slice(0, fd.row_limit);
  }

  const summaries: PipelineSummary[] = deriveSummaries(runs);

  return { width, height, formData: fd, runs, summaries };
}
