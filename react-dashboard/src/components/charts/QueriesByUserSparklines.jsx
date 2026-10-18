import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import { useAllUsers } from '../../hooks/useAllUsers';
import LoadingState from '../LoadingState';
import Pager from '../Pager';

const EMPTY_USER_IDS = [];
const PAGE_SIZE_OPTIONS = [5, 10, 15, 20];

// same data as QueriesByUser.jsx, but each user gets their own small silhouette
// instead of one combined chart where the peaks tangle together
function QueriesByUserSparklines({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const { displayName } = useUserFilter();
  const allUsers = useAllUsers();
  const [chartData, setChartData] = useState(null); // { months, countsByUser }
  const [rankedUsers, setRankedUsers] = useState([]);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize, setPageSize] = useState(5);
  const [error, setError] = useState(false);

  useEffect(() => {
    setPageIndex(0);
  }, [userIds]);

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
      .then((rows) => {
        const countsByUser = {};
        const totalByUser = {};
        userIds.forEach((u) => {
          countsByUser[u] = {};
          totalByUser[u] = 0;
        });

        const monthSet = new Set();
        rows.forEach(({ user_id, day, count }) => {
          const month = day.slice(0, 7); // group by month, not exact day
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
    return null;
  }

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!chartData || rankedUsers.length === 0) {
    return <LoadingState height="200px" />;
  }

  const pageStart = pageIndex * pageSize;
  const pagedUsers = rankedUsers.slice(pageStart, pageStart + pageSize);

  return (
    <div>
      {/* same layout as the other per-user charts: "Show" on the left, the title centered below it */}
      <div style={{ marginBottom: '0.5rem' }}>
        <Pager
          pageSize={pageSize}
          onPageSizeChange={setPageSize}
          pageIndex={pageIndex}
          onPageIndexChange={setPageIndex}
          totalItems={rankedUsers.length}
          pageSizeOptions={PAGE_SIZE_OPTIONS}
          label="Most active "
        />
      </div>
      <h3 style={{ color: colors.textOnPage, fontSize: '1rem', textAlign: 'center', margin: '0 0 0.75rem' }}>Activity Shape by User</h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, 180px)',
          justifyContent: 'center', // centers the whole block of tiles, not stretched edge-to-edge
          gap: '0.75rem'
        }}
      >
        {pagedUsers.map((user) => {
          const values = chartData.months.map((m) => chartData.countsByUser[user][m] || 0);
          const color = colors.categorical[allUsers.indexOf(user) % colors.categorical.length];

          const option = {
            grid: { left: 0, right: 0, top: 4, bottom: 0 },
            xAxis: { type: 'category', show: false, data: chartData.months },
            yAxis: { type: 'value', show: false },
            series: [
              {
                type: 'line',
                data: values,
                showSymbol: false,
                smooth: false,
                lineStyle: { width: 1.5, color },
                areaStyle: { color, opacity: 0.35 }
              }
            ]
          };

          return (
            <div
              key={user}
              style={{
                border: `1px solid ${colors.cardBorder}`,
                borderRadius: '10px',
                padding: '0.5rem 0.6rem 0.4rem',
                backgroundColor: `${colors.tileBorder}0a` // faint tint of the theme blue, works on light and dark cards
              }}
            >
              <Chart option={option} notMerge={true} style={{ height: '60px', width: '100%' }} />
              <div
                style={{
                  color: colors.textOnPage,
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  marginTop: '0.3rem',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
                title={displayName(user)}
              >
                {displayName(user)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default QueriesByUserSparklines;
