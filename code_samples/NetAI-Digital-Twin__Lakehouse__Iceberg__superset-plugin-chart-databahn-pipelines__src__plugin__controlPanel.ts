/**
 * controlPanel — defines the sidebar controls that appear in the Explore view
 * when this chart type is selected.
 *
 * Each section below maps to a logical group of user-configurable options.
 * Sections are intentionally comprehensive: trim once the real data model is
 * finalized.
 */
import { t } from '@superset-ui/core';
import { ControlPanelConfig } from '@superset-ui/chart-controls';

const config: ControlPanelConfig = {
  controlPanelSections: [
    // ── Display Mode ─────────────────────────────────────────────────────
    {
      label: t('Display'),
      expanded: true,
      controlSetRows: [
        [
          {
            name: 'display_mode',
            config: {
              type: 'SelectControl',
              label: t('Display Mode'),
              description: t(
                'Choose how pipeline data is visualized. ' +
                '"Summary Table" shows aggregated stats per pipeline; ' +
                '"Timeline" plots runs on a time axis; ' +
                '"Status Breakdown" shows a pie/donut of statuses; ' +
                '"Duration Histogram" buckets run durations; ' +
                '"Step Gantt" renders per-step timelines for a selected run.',
              ),
              default: 'summary_table',
              choices: [
                ['summary_table', t('Summary Table')],
                ['timeline', t('Run Timeline')],
                ['status_breakdown', t('Status Breakdown')],
                ['duration_histogram', t('Duration Histogram')],
                ['step_gantt', t('Step Gantt Chart')],
              ],
              renderTrigger: true,
            },
          },
        ],
      ],
    },

    // ── Filters ──────────────────────────────────────────────────────────
    {
      label: t('Filters'),
      expanded: true,
      controlSetRows: [
        [
          {
            name: 'pipeline_filter',
            config: {
              type: 'TextControl',
              label: t('Pipeline Name'),
              description: t(
                'Filter to a specific pipeline name or slug. Leave blank to show all pipelines.',
              ),
              default: '',
              renderTrigger: true,
            },
          },
        ],
        [
          {
            name: 'status_filter',
            config: {
              type: 'SelectControl',
              label: t('Status Filter'),
              description: t('Show only runs with this status.'),
              default: '',
              choices: [
                ['', t('All')],
                ['success', t('Success')],
                ['failure', t('Failure')],
                ['running', t('Running')],
                ['queued', t('Queued')],
                ['cancelled', t('Cancelled')],
                ['timeout', t('Timeout')],
                ['skipped', t('Skipped')],
              ],
              renderTrigger: true,
            },
          },
        ],
        [
          {
            name: 'lookback_hours',
            config: {
              type: 'SelectControl',
              label: t('Lookback Window'),
              description: t('Only display runs from the last N hours.'),
              default: 24,
              choices: [
                [1, t('1 hour')],
                [6, t('6 hours')],
                [12, t('12 hours')],
                [24, t('24 hours')],
                [72, t('3 days')],
                [168, t('7 days')],
                [720, t('30 days')],
              ],
              renderTrigger: true,
            },
          },
        ],
      ],
    },

    // ── Options ──────────────────────────────────────────────────────────
    {
      label: t('Options'),
      expanded: false,
      controlSetRows: [
        [
          {
            name: 'show_steps',
            config: {
              type: 'CheckboxControl',
              label: t('Show Step Details'),
              description: t('When enabled, per-step breakdowns are included where applicable.'),
              default: false,
              renderTrigger: true,
            },
          },
        ],
        [
          {
            name: 'row_limit',
            config: {
              type: 'SelectControl',
              label: t('Row Limit'),
              description: t('Maximum number of run rows to display.'),
              default: 100,
              choices: [
                [25, '25'],
                [50, '50'],
                [100, '100'],
                [250, '250'],
                [500, '500'],
                [1000, '1000'],
              ],
              renderTrigger: true,
            },
          },
        ],
      ],
    },
  ],
};

export default config;
