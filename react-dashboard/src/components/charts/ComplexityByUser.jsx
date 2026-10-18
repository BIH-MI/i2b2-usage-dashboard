import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import LoadingState from '../LoadingState';
import QueryLinkList from '../QueryLinkList';

const EMPTY_USER_IDS = [];
const MIN_QUERIES = 10; 
const VISIBLE_USERS = 15;

// average complexity score per user, highest first
function ComplexityByUser({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const { displayName } = useUserFilter();
  const [queries, setQueries] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return;
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/queries?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setQueries(rows))
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!queries) {
    return <LoadingState height="360px" />;
  }

  const unique = [...new Map(queries.map((q) => [q.query_master_id, q])).values()];
  const queriesByUser = {};
  unique.forEach((q) => {
    (queriesByUser[q.user_id] ||= []).push(q);
  });

  const users = Object.entries(queriesByUser)
    .filter(([, userQueries]) => userQueries.length >= MIN_QUERIES)
    .map(([userId, userQueries]) => {
      const total = userQueries.reduce((sum, q) => sum + q.complexity_score, 0);
      return { userId, queries: userQueries, average: Math.round((total / userQueries.length) * 10) / 10 };
    })
    .sort((a, b) => a.average - b.average); // lowest first

  if (users.length === 0) {
    return (
      <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>
        No user has {MIN_QUERIES} or more queries in this range.
      </p>
    );
  }

  const scroll = users.length > VISIBLE_USERS;

  const option = {
    title: { text: 'Query Complexity by User', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const user = users[params[0].dataIndex];
        return `<strong>${displayName(user.userId)}</strong><br/>Average complexity: ${user.average}<br/>Queries: ${user.queries.length}`;
      }
    },
    grid: { left: 10, right: scroll ? 50 : 40, top: 50, bottom: 30, containLabel: true },
    xAxis: {
      type: 'value',
      name: 'Average complexity score',
      nameLocation: 'middle',
      nameGap: 25,
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'category',
      data: users.map((u) => u.userId),
      axisLabel: { color: colors.textOnPage }
    },
    dataZoom: scroll
      ? [
          { type: 'slider', yAxisIndex: 0, right: 5, width: 15, startValue: users.length - VISIBLE_USERS, endValue: users.length - 1 },
          { type: 'inside', yAxisIndex: 0 }
        ]
      : [],
    series: [
      {
        type: 'bar',
        data: users.map((u) => u.average),
        itemStyle: { color: colors.categorical[0] },
        label: { show: true, position: 'right', color: colors.textOnPage, fontSize: 11 }
      }
    ]
  };

  const clicked = users.find((u) => u.userId === selectedUser);

  return (
    <div>
      <Chart
        option={option}
        notMerge
        style={{ height: '360px', width: '100%' }}
        onEvents={{ click: (params) => setSelectedUser(params.name) }}
      />

      {clicked && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={clicked.queries} title={displayName(clicked.userId)} />
        </div>
      )}
    </div>
  );
}

export default ComplexityByUser;
