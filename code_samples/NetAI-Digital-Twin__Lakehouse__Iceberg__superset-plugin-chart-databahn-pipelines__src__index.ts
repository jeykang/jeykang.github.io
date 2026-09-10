/**
 * Licensed under the Apache License, Version 2.0.
 *
 * superset-plugin-chart-databahn-pipelines
 *
 * Superset visualization plugin for monitoring Databahn pipeline / workflow
 * execution data.  This is a *template* — it renders placeholder UI with
 * simulated data so the visual structure can be evaluated before the real
 * Databahn data source is connected.
 */

export { DatabannPipelinesChartPlugin as default } from './plugin';
export { DatabannPipelinesChartPlugin } from './plugin';
export type {
  DatabannPipelinesFormData,
  DatabannPipelinesChartProps,
  PipelineRun,
  PipelineSummary,
  StepDetail,
} from './types';
