import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import { useAllUsers } from '../../hooks/useAllUsers';
import LoadingState from '../LoadingState';
import Pager from '../Pager';
import QueryLinkList from '../QueryLinkList';

const EMPTY_USER_IDS = [];

const PAGE_SIZE_OPTIONS = [5, 10, 15, 20];

function QueriesByUserOverlay({ startDate, endDate, userIds = EMPTY_USER_IDS, pageSize, onPageSizeChange }) {
  const { colors } = useTheme();
  const { displayName } = useUserFilter();
  const allUsers = useAllUsers();
  const [chartData, setChartData] = useState(null); // { months, countsByUser }
  const [rankedUsers, setRankedUsers] = useState([]); // userIds sorted by total query count, desc
  const [pageIndex, setPageIndex] = useState(0);
  const [hoveredUser, setHoveredUser] = useState(null); // drives endLabel visibility, see series map below
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedMonth, setSelectedMonth] = useState(null);
  const [monthQueries, setMonthQueries] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setPageIndex(0); // selection changed, back to page 1
    setSelectedUser(null);
    setSelectedMonth(null);
  }, [userIds]);

  useEffect(() => {
    if (!selectedUser || !selectedMonth) return;
    setMonthQueries(null);
    const [year, month] = selectedMonth.split('-').map(Number);
    const start = new Date(year, month - 1, 1); // 1st of the clicked month
    const end = new Date(year, month, 0); // last day of the clicked month
    const params = buildDateRangeParams(start, end, [selectedUser]);
    fetchWithTimeout(`/api/queries?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setMonthQueries(rows))
      .catch(() => setMonthQueries([]));
  }, [selectedUser, selectedMonth]);

  useEffect(() => {
    if (!startDate || !endDate || userIds.length === 0) {
      setChartData(null);
      setRankedUsers([]);
      return;
    }

    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/queries-by-user?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => { // rows = [{ user_id, day: "2025-08-14", count }, ...]
        const countsByUser = {}; // { user_id: { "2025-08": count, ... } }
        const totalByUser = {}; // user_id to total count, for ranking
        userIds.forEach((u) => {
          countsByUser[u] = {};
          totalByUser[u] = 0;
        });

        const monthSet = new Set();
        rows.forEach(({ user_id, day, count }) => {
          const month = day.slice(0, 7);
          monthSet.add(month);
          countsByUser[user_id][month] = (countsByUser[user_id][month] || 0) + count;
          totalByUser[user_id] += count;
        });

        const months = [...monthSet].sort();
        const ranked = [...userIds].sort((a, b) => totalByUser[b] - totalByUser[a]);

        setChartData({ months, countsByUser });
        setRankedUsers(ranked);
      })
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  if (userIds.length === 0) {
    return null; // nothing selected yet, UsersPage already shows a message for this
  }

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!chartData || rankedUsers.length === 0) {
    return <LoadingState height="380px" />; // selection made, data still in flight
  }

  const pageStart = pageIndex * pageSize;
  const pagedUsers = rankedUsers.slice(pageStart, pageStart + pageSize);

  const series = pagedUsers.map((user) => {
    const color = colors.categorical[allUsers.indexOf(user) % colors.categorical.length];
    return {
      name: user,
      type: 'line',
      smooth: false,
      lineStyle: { width: 1.5, color },
      areaStyle: { color, opacity: 0.35 }, // semi-transparent so overlapping areas don't fully hide each other
      emphasis: { focus: 'series' },
      endLabel: {
        show: hoveredUser === user,
        formatter: '{a}',
        color: colors.textOnPage,
        fontSize: 11
      },
      color,
      data: chartData.months.map((m) => chartData.countsByUser[user][m] || 0)
      // no `stack`, each user's area starts from its own 0
    };
  });

  const option = {
    title: { text: 'Query Activity by User (Overlaid)', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    legend: {
      type: 'scroll',
      data: series.map((s) => s.name),
      textStyle: { color: colors.textOnPage },
      pageTextStyle: { color: colors.textOnPage },
      pageIconColor: colors.tileBorder,
      pageIconInactiveColor: `${colors.tileBorder}55`,
      bottom: 0,
      left: 40,
      right: 40
    },
    grid: { top: 50, bottom: 60, right: 90 }, // room so endLabel (user name) isn't clipped at the right edge
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: chartData.months,
      axisLabel: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series
  };

  return (
    <div>
      <div style={{ marginBottom: '0.5rem' }}>
        <Pager
          pageSize={pageSize}
          onPageSizeChange={onPageSizeChange}
          pageIndex={pageIndex}
          onPageIndexChange={setPageIndex}
          totalItems={rankedUsers.length}
          pageSizeOptions={PAGE_SIZE_OPTIONS}
          label="Most active "
        />
      </div>
      <Chart
        option={option}
        notMerge={true}
        style={{ height: '380px', width: '100%' }}
        onEvents={{
          mouseover: (params) => {
            if (params.seriesName) setHoveredUser(params.seriesName);
          },
          mouseout: () => setHoveredUser(null),
          click: (params) => {
            setSelectedUser(params.seriesName);
            setSelectedMonth(params.name);
          }
        }}
      />

      {selectedUser && selectedMonth && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={monthQueries} title={`${displayName(selectedUser)} - ${selectedMonth}`} />
        </div>
      )}
    </div>
  );
}

export default QueriesByUserOverlay;
