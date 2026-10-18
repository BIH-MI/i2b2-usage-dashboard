import Chart from '../Chart';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../../theme';
import { formatDuration } from '../../utils/formatDuration';

// x = how many constraints (date/value/occurrence) a query has, y = how long it took
function ConstraintsScatter({ queries }) {
  const { colors } = useTheme();
  const navigate = useNavigate();

  const completed = queries.filter((q) => q.duration != null && q.duration > 0);

  const toPoint = (q) => ({
    value: [q.num_constraints, q.duration],
    name: q.name,
    status_name: q.status_name,
    temporal: q.temporal_query_constraints > 0,
    query_master_id: q.query_master_id
  });

  const withTemporal = completed.filter((q) => q.temporal_query_constraints > 0).map(toPoint);
  const withoutTemporal = completed.filter((q) => q.temporal_query_constraints === 0).map(toPoint);

  const tooltipFormatter = (p) =>
    `<strong>${p.data.name}</strong><br/>Constraints: ${p.data.value[0]}<br/>Duration: ${formatDuration(p.data.value[1])}<br/>Temporal ordering: ${p.data.temporal ? 'yes' : 'no'}<br/>Status: ${p.data.status_name}`;

  const option = {
    title: { text: 'Constraints vs Duration', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'item',
      confine: true, // keeps the tooltip inside the chart area instead of running off past the screen edge
      formatter: tooltipFormatter,
      extraCssText: 'max-width: 280px; white-space: normal;' // wraps long query names instead of one long line
    },
    legend: {
      data: ['With temporal ordering', 'Without temporal ordering'],
      textStyle: { color: colors.textOnPage },
      bottom: 0 // below the chart, at the top it ran into the title on narrow cards
    },
    grid: { top: 60, bottom: 60 },
    xAxis: {
      type: 'value',
      name: 'Constraints',
      minInterval: 1,
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'log',
      name: 'Duration (s, log scale)',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'inside', yAxisIndex: 0 }
    ],
    series: [
      {
        name: 'With temporal ordering',
        type: 'scatter',
        symbolSize: 12,
        itemStyle: { color: colors.categorical[1] },
        data: withTemporal
      },
      {
        name: 'Without temporal ordering',
        type: 'scatter',
        symbolSize: 12,
        itemStyle: { color: colors.categorical[0] },
        data: withoutTemporal
      }
    ]
  };

  return completed.length === 0 ? (
    <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No completed queries in this range.</p>
  ) : (
    <>
      <p style={{ color: colors.textOnPage, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
        Click a point to open that query's full details.
      </p>
      <Chart
        option={option}
        style={{ height: '360px', width: '100%' }}
        onEvents={{
          click: (params) => {
            if (params.data?.query_master_id) navigate(`/queries/${params.data.query_master_id}`);
          }
        }}
      />
    </>
  );
}

export default ConstraintsScatter;
