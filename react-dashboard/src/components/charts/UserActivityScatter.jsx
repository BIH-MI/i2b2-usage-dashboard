import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import LoadingState from '../LoadingState';
import QueryLinkList from '../QueryLinkList';
const EMPTY_USER_IDS = [];


// one point per user, x/y are the backend's activity_score inputs, the score itself is tooltip-only
function UserActivityScatter({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const { displayName } = useUserFilter();
  const [rows, setRows] = useState(null); // [{ user_id, query_count, valid_sessions, activity_score }, ...]
  const [userQueries, setUserQueries] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return;
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/user-activity?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((data) => setRows(data))
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  useEffect(() => {
    if (!selectedUser) return;
    const params = buildDateRangeParams(startDate, endDate, [selectedUser]);
    fetchWithTimeout(`/api/queries?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((data) => setUserQueries(data))
      .catch(() => setUserQueries([]));
  }, [startDate, endDate, selectedUser]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!rows) {
    return <LoadingState height="400px" />;
  }

  const selectedRow = rows.find((r) => r.user_id === selectedUser); // same data already shown in the tooltip on hover

  const option = {
    title: { text: 'User Activity', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const { user_id, query_count, valid_sessions, activity_score } = params.data;
        return `<strong>${user_id}</strong><br/>Queries: ${query_count}<br/>Valid sessions: ${valid_sessions}<br/>Activity score: ${activity_score}`;
      }
    },
    xAxis: {
      type: 'value',
      name: 'Valid sessions',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    // most users cluster near the origin, scroll to zoom in, drag to pan
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'inside', yAxisIndex: 0 }
    ],
    series: [
      {
        type: 'scatter',
        symbolSize: 12,
        itemStyle: { color: colors.categorical[0] },
        data: rows.map((r) => ({
          value: [r.valid_sessions, r.query_count],
          user_id: r.user_id,
          query_count: r.query_count,
          valid_sessions: r.valid_sessions,
          activity_score: r.activity_score
        }))
      }
    ]
  };

  return (
    <div>
      <p style={{ color: colors.textOnPage, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
        Scroll to zoom into crowded areas, drag to pan.
      </p>
      <Chart
        option={option}
        style={{ height: '400px', width: '100%' }}
        onEvents={{ click: (params) => setSelectedUser(params.data.user_id) }}
      />

      {selectedUser && (
        <div style={{ marginTop: '1rem' }}>
          {selectedRow && (
            <p style={{ color: `${colors.textOnPage}99`, fontSize: '0.8rem', marginBottom: '0.5rem' }}>
              Valid sessions: {selectedRow.valid_sessions} · Queries: {selectedRow.query_count} · Activity score: {selectedRow.activity_score}
            </p>
          )}

          <QueryLinkList queries={userQueries} title={displayName(selectedUser)} />
        </div>
      )}
    </div>
  );
}

export default UserActivityScatter;
