# superset-plugin-chart-databahn-pipelines

> **Status: Template / Scaffold — renders simulated data; no live Databahn connection yet.**

A custom Apache Superset visualization plugin for monitoring **Databahn
pipeline / workflow** execution data.  It provides five display modes out of
the box and is designed to be connected to a Databahn data source once access
to the system's database or API is available.

---

## Table of Contents

1. [What This Plugin Does](#what-this-plugin-does)
2. [Display Modes](#display-modes)
3. [Data Model (What We Plan to Visualize)](#data-model)
4. [Plugin Architecture](#plugin-architecture)
5. [File Layout](#file-layout)
6. [How to Install in Superset](#how-to-install-in-superset)
7. [Development Workflow](#development-workflow)
8. [What's Needed From the Databahn Side](#whats-needed-from-databahn)
9. [Integration Paths](#integration-paths)
10. [Open Questions](#open-questions)

---

## What This Plugin Does

The plugin registers a new chart type called **"Databahn Pipelines"** in the
Superset Explore view.  When selected, users can configure filters and display
modes via the sidebar control panel and see pipeline monitoring data visualized
in the chart area.

Because we don't yet have access to the real Databahn database or API, the
plugin ships with a **deterministic simulated-data generator** that produces
realistic-looking pipeline runs, so the UI can be evaluated immediately.

---

## Display Modes

| Mode | Description |
|---|---|
| **Summary Table** | One row per pipeline: total runs, success/failure counts, success rate, avg/p90 duration, current running count, last status. |
| **Run Timeline** | Flat table of individual runs sorted newest-first, showing run ID, pipeline, timestamps, status, trigger type, user, and compute environment. |
| **Status Breakdown** | Stacked horizontal bars per pipeline, colour-coded by terminal status (success, failure, cancelled, timeout). |
| **Duration Histogram** | Bucketed bar chart of run durations across all matching runs. |
| **Step Gantt Chart** | Per-step duration breakdown for the most recent run that contains step-level data. |

---

## Data Model

The type definitions in `src/types.ts` are intentionally broad.  They cover
every category of data that is commonly useful for pipeline monitoring:

### Per-Run Record (`PipelineRun`)

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Unique execution identifier |
| `pipeline_name` | string | Pipeline slug / key |
| `pipeline_label` | string? | Human-readable name |
| `pipeline_category` | string? | Grouping (e.g. "annotation", "perception") |
| `triggered_at` | ISO-8601 | When the run was enqueued |
| `started_at` | ISO-8601? | When execution started |
| `finished_at` | ISO-8601? | When execution ended |
| `duration_seconds` | number? | Wall-clock duration |
| `status` | enum | `success` · `failure` · `running` · `queued` · `cancelled` · `timeout` · `skipped` |
| `error_message` | string? | Error text (failures only) |
| `error_type` | string? | Error classification |
| `retry_count` | number? | Auto-retry attempts consumed |
| `triggered_by` | string? | User / service account |
| `trigger_type` | enum? | `manual` · `scheduled` · `event` · `api` · `webhook` |
| `pipeline_version` | string? | Git/version ref of the pipeline definition |
| `run_metadata` | object? | Arbitrary key-value pairs |
| `input_datasets` | string[]? | Input dataset identifiers |
| `output_datasets` | string[]? | Output dataset identifiers |
| `compute_environment` | string? | Compute target label |
| `resource_usage` | object? | CPU-seconds, peak-memory-MB, GPU-hours, … |
| `steps` | StepDetail[]? | Per-step breakdown |

### Per-Step Record (`StepDetail`)

| Field | Type | Description |
|---|---|---|
| `step_name` | string | Step label |
| `step_order` | number | Sequence number |
| `status` | enum | `success` · `failure` · `running` · `skipped` |
| `started_at` / `finished_at` | ISO-8601? | Timestamps |
| `duration_seconds` | number? | Step wall-clock time |
| `error_message` | string? | Step-level error |
| `step_metrics` | object? | Numeric KPIs emitted by the step |

### Pipeline Summary (`PipelineSummary`)

Derived from runs; one record per pipeline:

- Total / successful / failed / cancelled / timeout / running / queued counts
- Success rate, avg duration, p50/p90/p99 duration
- Last run timestamp & status
- Throughput (runs/hour)

---

## Plugin Architecture

Superset viz plugins follow a fixed contract:

```
┌─────────────────────────────────────────────────────────────┐
│  Superset Frontend                                          │
│                                                             │
│  1. controlPanel.ts  → sidebar controls (user picks mode,   │
│                        filters, options)                     │
│  2. buildQuery.ts    → constructs the QueryContext sent to   │
│                        the backend SQL engine                │
│  3. transformProps.ts→ maps raw query results to the props   │
│                        the React component expects           │
│  4. Component.tsx    → renders the visualization             │
│  5. index.ts (plugin)→ registers metadata, thumbnail, the   │
│                        above hooks into Superset's registry  │
└─────────────────────────────────────────────────────────────┘
```

Each file has a single responsibility:

| File | Role |
|---|---|
| `src/plugin/controlPanel.ts` | Defines sidebar UI: dropdowns, checkboxes, text filters |
| `src/plugin/buildQuery.ts` | Translates form data into the SQL/API query payload |
| `src/plugin/transformProps.ts` | Converts raw backend rows into typed chart props; also falls back to simulated data |
| `src/DatabannPipelinesChart.tsx` | Pure React component — picks one of five sub-renderers based on `display_mode` |
| `src/simulatedData.ts` | Deterministic fake data generator (remove once real source is connected) |
| `src/types.ts` | TypeScript interfaces shared across all files |
| `src/plugin/index.ts` | `ChartPlugin` subclass that ties everything together |

---

## File Layout

```
superset-plugin-chart-databahn-pipelines/
├── package.json
├── tsconfig.json
├── jest.config.js
├── .gitignore
├── README.md                        ← this file
├── types/
│   └── external.d.ts                ← image import declarations
├── src/
│   ├── index.ts                     ← public entry point
│   ├── types.ts                     ← PipelineRun, PipelineSummary, etc.
│   ├── simulatedData.ts             ← fake data generator
│   ├── DatabannPipelinesChart.tsx    ← main React component
│   ├── images/
│   │   └── thumbnail.png            ← chart picker thumbnail
│   └── plugin/
│       ├── index.ts                 ← ChartPlugin class
│       ├── buildQuery.ts            ← query builder
│       ├── controlPanel.ts          ← sidebar controls
│       └── transformProps.ts        ← data transformer
└── test/
    ├── index.test.ts
    ├── __mocks__/
    │   └── mockExportString.js
    └── plugin/
        ├── buildQuery.test.ts
        ├── transformProps.test.ts
        └── simulatedData.test.ts
```

---

## How to Install in Superset

### Option A — Local development (npm link)

```bash
# 1. Build the plugin
cd superset-plugin-chart-databahn-pipelines
npm install --force
npm run build

# 2. Link into Superset's frontend
cd /path/to/superset/superset-frontend
npm i -S /path/to/superset-plugin-chart-databahn-pipelines
```

Then edit `superset-frontend/src/visualizations/presets/MainPreset.js`:

```js
import { DatabannPipelinesChartPlugin } from 'superset-plugin-chart-databahn-pipelines';

// …inside the plugins array:
new DatabannPipelinesChartPlugin().configure({ key: 'ext-databahn-pipelines' }),
```

Run `npm run dev-server` to see the chart in the Explore view.

### Option B — Docker image (production)

1. Publish the plugin to npm (or copy source into the build context).
2. In the Superset Dockerfile, add:
   ```dockerfile
   RUN cd /app/superset-frontend && npm i -S superset-plugin-chart-databahn-pipelines
   ```
3. Patch `MainPreset.js` as above.
4. Rebuild: `docker build -t apache/superset:databahn --target lean .`
5. Update `docker-compose.yml` to use the new image tag.

---

## Development Workflow

```bash
cd superset-plugin-chart-databahn-pipelines

# Install deps
npm install --force

# Watch mode — rebuilds on every change
npm run dev

# Run tests
npm test

# One-shot build
npm run build
```

---

## What's Needed From the Databahn Side

Before the plugin can display real data, we need answers to these questions
(see also [Open Questions](#open-questions)):

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | **Database type** — Postgres, MySQL, ClickHouse, SQLite, proprietary? | Determines whether Superset can connect natively or needs a custom DB engine driver. |
| 2 | **API availability** — Does Databahn expose a REST / gRPC API for run history? | If SQL access isn't possible, we'd need a Superset "dynamic datasource" or a lightweight ETL bridge. |
| 3 | **Schema / table layout** — What tables store runs, steps, pipelines? Column names? | Needed to rewrite `buildQuery.ts` and `transformProps.ts`. |
| 4 | **Authentication** — How does the cluster authenticate external clients? | Needed for connection-string or API-token configuration. |
| 5 | **Data volume** — Rough order-of-magnitude of runs per day? | Affects whether we need server-side aggregation or can pull raw rows. |
| 6 | **Real-time needs** — Should the dashboard auto-refresh? What latency is acceptable? | Determines caching strategy and refresh interval. |
| 7 | **Multi-tenancy** — Are pipelines scoped to teams / projects? | May need tenant-aware filters in the control panel. |

---

## Integration Paths

Depending on what Databahn exposes, there are three likely integration
strategies:

### Path 1 — Direct SQL Connection

If Databahn stores run history in a SQL-compatible database (Postgres, MySQL,
etc.), Superset can connect to it as a regular "database" via the Admin →
Database Connections UI.  The plugin's `buildQuery.ts` would emit raw SQL
against the Databahn tables.

**Pros:** Simplest path, native Superset support, full SQL flexibility.
**Cons:** Requires network access from Superset to Databahn's DB; read-only
user with appropriate permissions needed.

### Path 2 — REST/gRPC API → Staging Table

A lightweight ETL job (Python script or Airflow DAG) periodically pulls run
data from Databahn's API and writes it into the lakehouse (e.g. an Iceberg
table via Spark or Trino).  Superset then queries the Iceberg table through
Trino, which is already configured in this stack.

**Pros:** Decouples Superset from Databahn's internal DB; data lands in the
lakehouse alongside other datasets; works even if Databahn's DB is
non-standard.
**Cons:** Slightly higher latency (polling interval); extra moving part.

### Path 3 — Custom Superset Datasource

Superset supports custom datasource plugins (Python-side).  A thin adapter
class could translate Superset query objects into Databahn API calls on the
fly, returning the result as a dataframe.

**Pros:** No staging table needed; real-time.
**Cons:** More complex to implement and maintain; less common pattern.

---

## Open Questions

- [ ] What database engine does Databahn use?
- [ ] Is there a public or internal API for querying run history?
- [ ] What are the exact table / column names for runs, steps, and pipelines?
- [ ] Does Databahn support webhooks that could push events to the lakehouse?
- [ ] Are there existing Grafana / Prometheus metrics endpoints we could reuse?
- [ ] What authentication mechanism does the local cluster deployment use?
- [ ] Should the Superset charts be embedded in the Databahn UI, or vice-versa?
- [ ] Which of the five display modes are highest priority for the team?
- [ ] Is step-level granularity consistently available across all pipeline types?
- [ ] Are there additional KPIs (cost, data volume processed, model accuracy)
  that should be added to the data model?
