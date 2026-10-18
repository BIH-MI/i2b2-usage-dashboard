import { useState } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import QueryLinkList from '../QueryLinkList';
import { statusColorMap } from '../../utils/statusColors';

// derived entirely from the queries QueriesPage already fetched, no fetching of its own
function QueryStatus({ queries }) {
  const { colors } = useTheme();
  const STATUS_COLOR_MAP = statusColorMap(colors);
  const [selectedStatus, setSelectedStatus] = useState(null);

  const counts = {};
  queries.forEach((q) => {
    counts[q.status_name] = (counts[q.status_name] || 0) + 1;
  });

  // FINISHED usually dwarfs the others, excluded so the problem statuses stay comparable
  const statusTotals = Object.entries(counts)
    .filter(([status_name]) => status_name !== 'FINISHED')
    .map(([status_name, count]) => ({ status_name, count }));

  const matchingQueries = selectedStatus ? queries.filter((q) => q.status_name === selectedStatus) : [];

  const option = {
    title: { text: 'Queries by Status', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    grid: { bottom: 70 }, // room for the rotated labels so they don't get clipped
    xAxis: {
      type: 'category',
      data: statusTotals.map((s) => s.status_name),
      axisLabel: { color: colors.textOnPage, interval: 0, rotate: 30 }, // interval: 0 never auto-hides labels
      triggerEvent: true // labels fire click events too, same as the bars
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 44,
        label: { show: true, position: 'top', color: colors.textOnPage, fontSize: 11 },
        data: statusTotals.map((s) => ({
          value: s.count,
          // status colors keep their meaning, rounded tops like the other bar charts
          itemStyle: { color: STATUS_COLOR_MAP[s.status_name] || colors.status.neutral, borderRadius: [4, 4, 0, 0] }
        }))
      }
    ]
  };

  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1.5rem', width: '100%' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ color: colors.textOnPage, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
          Click a bar or label to see which queries had that status.
        </p>
        <Chart
          option={option}
          style={{ height: '320px', width: '100%' }}
          onEvents={{
            click: (params) => {
              // bar click gives params.name, axisLabel click gives params.value
              setSelectedStatus(params.componentType === 'xAxis' ? params.value : params.name);
            }
          }}
        />
      </div>

      {selectedStatus && (
        <div style={{ flex: 1, minWidth: 0, marginTop: '1.6rem' }}>
          <div style={{ maxHeight: '320px', overflowY: 'auto' }}>
            <QueryLinkList queries={matchingQueries} title={selectedStatus} />
          </div>
        </div>
      )}
    </div>
  );
}

export default QueryStatus;
