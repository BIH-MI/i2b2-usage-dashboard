import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import LoadingState from '../LoadingState';
import QueryLinkList from '../QueryLinkList';

const EMPTY_USER_IDS = [];

// userIds is optional, with no selection it's the total across all users
function QueriesOverTime({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const [chartData, setChartData] = useState(null);
  const [selectedMonth, setSelectedMonth] = useState(null); // which month bucket was clicked, if any, "2025-08"
  const [monthQueries, setMonthQueries] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return; // date-range fetch hasn't resolved yet
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/daily-count?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => { // rows = [{ day: "2025-08-14", count: 3 }, ...], already grouped by the backend
        const counts = {}; // number of queries per month, keyed "2025-08"

        rows.forEach(({ day, count }) => {
          const month = day.slice(0, 7); // first 7 chars of "2025-08-14" is the month bucket
          counts[month] = (counts[month] || 0) + count;
        });

        const months = Object.keys(counts).sort();
        const values = months.map((m) => counts[m]);

        setChartData({ months, values });
      })
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  useEffect(() => {
    if (!selectedMonth) return;
    setMonthQueries(null);
    const [year, month] = selectedMonth.split('-').map(Number);
    const start = new Date(year, month - 1, 1); // 1st of the clicked month
    const end = new Date(year, month, 0); // last day of the clicked month
    const params = buildDateRangeParams(start, end, userIds);
    fetchWithTimeout(`/api/queries?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setMonthQueries(rows))
      .catch(() => setMonthQueries([]));
  }, [selectedMonth, userIds]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!chartData) {
    return <LoadingState height="350px" />;
  }

  const option = {
    title: { text: 'Queries Over Time', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: chartData.months,
      axisLabel: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        data: chartData.values,
        type: 'line',
        smooth: true
      }
    ]
  };

  return (
    <div>
      <Chart
        option={option}
        style={{ height: '350px', width: '100%' }}
        onEvents={{ click: (params) => setSelectedMonth(params.name) }}
      />

      {selectedMonth && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={monthQueries} title={selectedMonth} />
        </div>
      )}
    </div>
  );
}

export default QueriesOverTime;
