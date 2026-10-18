import Chart from '../Chart';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../../theme';
import { formatDuration } from '../../utils/formatDuration';

// x = how many distinct concepts a query touched, y = how long it took
function QueryDurationScatter({ queries }) {
  const { colors } = useTheme();
  const navigate = useNavigate();

  // still-running queries (duration null) and instant ones (0s breaks the log axis) are left out
  const points = queries
    .filter((q) => q.duration != null && q.duration > 0)
    .map((q) => ({
      value: [new Set(q.concepts).size, q.duration], // a Set drops duplicate concepts within one query
      name: q.name,
      status_name: q.status_name,
      query_master_id: q.query_master_id
    }));

  const option = {
    title: { text: 'Concepts Touched vs Duration', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'item',
      confine: true, // keeps the tooltip inside the chart area
      formatter: (p) => // p.data is the object built in `points` above
        `<strong>${p.data.name}</strong><br/>Concepts: ${p.data.value[0]}<br/>Duration: ${formatDuration(p.data.value[1])}<br/>Status: ${p.data.status_name}`,
      extraCssText: 'max-width: 280px; white-space: normal;' // wraps long query names instead of one long line
    },
    grid: { top: 75, bottom: 70 }, // top: clears the title from the y-axis name; bottom: room for the x-axis name
    xAxis: {
      type: 'value',
      name: 'Distinct concepts touched',
      nameLocation: 'center', // centered below the axis instead of squeezed at the end
      nameGap: 28,
      minInterval: 1, // grid lines only land on whole numbers, concept count can't be 2.5
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'log', // linear would squash fast queries into one unreadable cluster near zero
      name: 'Duration (s, log scale)',
      nameLocation: 'center', // centered along the axis instead of stacked at the top, next to the title
      nameGap: 50, // clears the log-scale tick labels (which can be wide)
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'inside', yAxisIndex: 0 }
    ],
    series: [
      {
        type: 'scatter',
        symbolSize: 12,
        itemStyle: { color: colors.categorical[0] },
        data: points
      }
    ]
  };

  return points.length === 0 ? (
    <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No completed queries in this range.</p>
  ) : (
    <Chart
      option={option}
      style={{ height: '320px', width: '100%' }}
      onEvents={{
        click: (params) => {
          if (params.data?.query_master_id) navigate(`/queries/${params.data.query_master_id}`);
        }
      }}
    />
  );
}

export default QueryDurationScatter;
