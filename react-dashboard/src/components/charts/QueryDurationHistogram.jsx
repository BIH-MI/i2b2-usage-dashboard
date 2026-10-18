import Chart from '../Chart';
import { useTheme } from '../../theme';
import { formatDuration } from '../../utils/formatDuration';

// log-scale buckets from 0.01s up, each bucket covers half a power of ten
const MIN_EXP = -2;
const STEP = 0.5;

// formatDuration shows 0.01s as "0.0s", so short buckets get one significant digit
function bucketLabel(seconds) {
  return seconds < 1 ? `${Number(seconds.toPrecision(1))}s` : formatDuration(seconds);
}

// how all query durations are spread, with the bar of this query highlighted
function QueryDurationHistogram({ queries, duration }) {
  const { colors } = useTheme();

  const unique = [...new Map(queries.map((q) => [q.query_master_id, q])).values()];
  const durations = unique.map((q) => q.duration).filter((d) => d !== null && d > 0);
  if (durations.length === 0 || duration === null) return null;

  const bucketOf = (d) => Math.max(0, Math.floor((Math.log10(d) - MIN_EXP) / STEP));
  const bucketCount = Math.max(...durations.map(bucketOf), bucketOf(Math.max(duration, 0.01))) + 1;
  const counts = new Array(bucketCount).fill(0);
  durations.forEach((d) => { counts[bucketOf(d)] += 1; });
  const current = bucketOf(Math.max(duration, 0.01));

  const labels = counts.map((_, i) => bucketLabel(10 ** (MIN_EXP + i * STEP)));

  const option = {
    title: { text: 'Duration Compared to All Queries', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const i = params[0].dataIndex;
        const to = i + 1 < labels.length ? bucketLabel(10 ** (MIN_EXP + (i + 1) * STEP)) : '…';
        return `${labels[i]} - ${to}<br/>${counts[i]} queries${i === current ? '<br/><b>this query is here</b>' : ''}`;
      }
    },
    grid: { left: 10, right: 20, top: 70, bottom: 10, containLabel: true },
    xAxis: { type: 'category', data: labels, name: 'from', axisLabel: { color: colors.textOnPage }, nameTextStyle: { color: colors.textOnPage } },
    yAxis: { type: 'value', name: 'Queries', axisLabel: { color: colors.textOnPage }, nameTextStyle: { color: colors.textOnPage } },
    series: [
      {
        type: 'bar',
        barCategoryGap: '15%',
        data: counts.map((c, i) => ({
          value: c,
          itemStyle: { color: i === current ? colors.status.serious : `${colors.tileBorder}88`, borderRadius: [3, 3, 0, 0] }
        })),
        markPoint: {
          symbol: 'pin',
          symbolSize: 40,
          itemStyle: { color: colors.status.serious },
          label: { formatter: 'this', fontSize: 10 },
          data: [{ coord: [current, counts[current]] }]
        }
      }
    ]
  };

  return <Chart option={option} style={{ height: '300px', width: '100%' }} />;
}

export default QueryDurationHistogram;
