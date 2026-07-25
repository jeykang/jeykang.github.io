import { generateSimulatedRuns, deriveSummaries } from '../../src/simulatedData';

describe('simulatedData', () => {
  it('generates the expected number of runs', () => {
    const runs = generateSimulatedRuns(50);
    expect(runs).toHaveLength(50);
  });

  it('every run has required fields', () => {
    const runs = generateSimulatedRuns(20);
    for (const r of runs) {
      expect(r.run_id).toBeDefined();
      expect(r.pipeline_name).toBeDefined();
      expect(r.triggered_at).toBeDefined();
      expect(r.status).toBeDefined();
    }
  });

  it('produces deterministic output (same seed)', () => {
    const a = generateSimulatedRuns(10);
    const b = generateSimulatedRuns(10);
    expect(a.map(r => r.run_id)).toEqual(b.map(r => r.run_id));
  });

  it('deriveSummaries groups runs by pipeline', () => {
    const runs = generateSimulatedRuns(80);
    const summaries = deriveSummaries(runs);
    expect(summaries.length).toBeGreaterThan(0);
    for (const s of summaries) {
      expect(s.total_runs).toBeGreaterThan(0);
      expect(s.success_rate).toBeGreaterThanOrEqual(0);
      expect(s.success_rate).toBeLessThanOrEqual(1);
    }
  });
});
