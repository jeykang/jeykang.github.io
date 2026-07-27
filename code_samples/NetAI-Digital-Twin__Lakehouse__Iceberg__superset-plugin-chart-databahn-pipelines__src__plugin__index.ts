/**
 * Plugin entry point — registers the chart with Superset's plugin system.
 */
import { t, ChartMetadata, ChartPlugin } from '@superset-ui/core';
import buildQuery from './buildQuery';
import controlPanel from './controlPanel';
import transformProps from './transformProps';
import thumbnail from '../images/thumbnail.png';

export class DatabannPipelinesChartPlugin extends ChartPlugin {
  constructor() {
    const metadata = new ChartMetadata({
      description: t(
        'Visualize Databahn pipeline / workflow execution data — run history, ' +
        'success & failure rates, durations, step-level breakdowns, and more. ' +
        'Supports multiple display modes: summary table, run timeline, status ' +
        'breakdown pie chart, duration histogram, and step Gantt chart.',
      ),
      name: t('Databahn Pipelines'),
      thumbnail,
      tags: [
        t('Databahn'),
        t('Pipeline'),
        t('Workflow'),
        t('Monitoring'),
        t('Operations'),
      ],
    });

    super({
      buildQuery,
      controlPanel,
      loadChart: () => import('../DatabannPipelinesChart'),
      metadata,
      transformProps,
    });
  }
}
