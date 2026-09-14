import { DatabannPipelinesChartPlugin } from '../src';

describe('DatabannPipelinesChartPlugin', () => {
  it('exports the plugin class', () => {
    expect(DatabannPipelinesChartPlugin).toBeDefined();
  });

  it('can be instantiated', () => {
    const plugin = new DatabannPipelinesChartPlugin();
    expect(plugin).toBeDefined();
  });
});
