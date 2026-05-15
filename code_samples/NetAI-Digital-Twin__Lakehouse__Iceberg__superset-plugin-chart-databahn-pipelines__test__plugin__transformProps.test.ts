import transformProps from '../../src/plugin/transformProps';

// Minimal mock of ChartProps
const makeMockChartProps = (overrides: Record<string, any> = {}) => ({
  width: 800,
  height: 600,
  formData: {
    display_mode: 'summary_table',
    ...overrides.formData,
  },
  queriesData: overrides.queriesData ?? [{ data: [] }],
});

describe('transformProps', () => {
  it('returns simulated data when query result is empty', () => {
    const result = transformProps(makeMockChartProps() as any);
    expect(result.runs.length).toBeGreaterThan(0);
    expect(result.summaries.length).toBeGreaterThan(0);
  });

  it('passes through width and height', () => {
    const result = transformProps(makeMockChartProps() as any);
    expect(result.width).toBe(800);
    expect(result.height).toBe(600);
  });

  it('applies pipeline_filter', () => {
    const result = transformProps(makeMockChartProps({
      formData: { display_mode: 'timeline', pipeline_filter: 'lidar' },
    }) as any);
    for (const r of result.runs) {
      expect(r.pipeline_name.toLowerCase()).toContain('lidar');
    }
  });

  it('applies status_filter', () => {
    const result = transformProps(makeMockChartProps({
      formData: { display_mode: 'timeline', status_filter: 'failure' },
    }) as any);
    for (const r of result.runs) {
      expect(r.status).toBe('failure');
    }
  });

  it('respects row_limit', () => {
    const result = transformProps(makeMockChartProps({
      formData: { display_mode: 'timeline', row_limit: 5 },
    }) as any);
    expect(result.runs.length).toBeLessThanOrEqual(5);
  });
});
