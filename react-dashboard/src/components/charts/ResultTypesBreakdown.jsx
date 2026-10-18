import Chart from '../Chart';
import { useTheme } from '../../theme';


function ResultTypesBreakdown({ queries }) {
  const { colors } = useTheme();

  const withResults = queries.filter((q) => q.results.length > 0); // some queries have no results

  const counts = {}; // keyed by result_type_name, value is how many queries requested it
  withResults.forEach((q) => {
    new Set(q.results.map((r) => r.result_type_name)).forEach((typeName) => {
      counts[typeName] = (counts[typeName] || 0) + 1;
    });
  });

  const entries = Object.entries(counts)
    .map(([name, count]) => ({ name, count, pct: (count / withResults.length) * 100 }))
    .sort((a, b) => b.count - a.count);

  const option = {
    title: { text: 'Result Types Requested', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0];
        const e = entries[p.dataIndex];
        return `<strong>${e.name}</strong><br/>${e.count} of ${withResults.length} queries (${e.pct.toFixed(1)}%)`;
      }
    },
    grid: { bottom: 90 },
    xAxis: {
      type: 'category',
      data: entries.map((e) => e.name),
      axisLabel: { color: colors.textOnPage, interval: 0, rotate: 20 }
    },
    yAxis: {
      type: 'value',
      name: '% of queries with a result',
      max: 100,
      axisLabel: { color: colors.textOnPage, formatter: '{value}%' },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 44,
        // one measure, so one color. Rounded tops, value on top of each bar
        itemStyle: { color: colors.categorical[2], borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'top', color: colors.textOnPage, fontSize: 11, formatter: (p) => `${p.value.toFixed(0)}%` },
        data: entries.map((e) => e.pct)
      }
    ]
  };

  return entries.length === 0 ? (
    <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No results in this range.</p>
  ) : (
    <Chart option={option} style={{ height: '360px', width: '100%' }} />
  );
}

export default ResultTypesBreakdown;
