import { useState, useEffect } from 'react';
import DateRangeFilter from '../components/DateRangeFilter';
import ChartCard from '../components/ChartCard';
import PageTitle from '../components/PageTitle';
import OverallNumbers from '../components/charts/OverallNumbers';
import QueriesOverTime from '../components/charts/QueriesOverTime';
import UsageOverTime from '../components/charts/UsageOverTime';
import ActivityCalendar from '../components/charts/ActivityCalendar';
import QueriesByUserViews from '../components/charts/QueriesByUserViews';
import QueriesByUserSparklines from '../components/charts/QueriesByUserSparklines';
import UserActivityScatter from '../components/charts/UserActivityScatter';
import LoginStatus from '../components/charts/LoginStatus';
import ComplexityByUser from '../components/charts/ComplexityByUser';
import { useDateRange } from '../context/DateRangeContext';
import { useUserFilter } from '../context/UserFilterContext';
import { buildDateRangeParams } from '../utils/dateRangeParams';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';
import { useTheme } from '../theme';

const EMPTY_USER_IDS = [];

function UsersPage() {
  const { colors } = useTheme();
  const { startDate, endDate, dataRange, error: dateRangeError, setStartDate, setEndDate, applyShortcut, showAllTime } = useDateRange();
  const { selectedUsers } = useUserFilter();
  const [activeUsers, setActiveUsers] = useState(EMPTY_USER_IDS);

  // fallback for the per-user charts: everyone with activity in this range, the charts page through them
  useEffect(() => {
    if (selectedUsers.length > 0 || !startDate || !endDate) return;
    const params = buildDateRangeParams(startDate, endDate);
    fetchWithTimeout(`/api/user-activity?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => {
        setActiveUsers(rows.map((r) => r.user_id)); // the charts sort them by query count themselves
      })
      .catch(() => setActiveUsers(EMPTY_USER_IDS));
  }, [startDate, endDate, selectedUsers]);

  // aggregate charts read selectedUsers directly, empty already means "all users" to them.
  // per-user-line charts need an explicit list, so they fall back to all active users
  const perUserChartIds = selectedUsers.length > 0 ? selectedUsers : activeUsers;

  return (
    <div style={{ padding: '2rem' }}>
      <PageTitle title="Users" />

      <DateRangeFilter
        startDate={startDate}
        endDate={endDate}
        minDate={dataRange?.min}
        onStartChange={setStartDate}
        onEndChange={setEndDate}
        onApplyShortcut={applyShortcut}
        onAllTime={showAllTime}
      />

      {dateRangeError ? (
        <div style={{ marginTop: '1.5rem' }}>
          <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>
        </div>
      ) : (
        <>
      <div style={{ marginTop: '1.5rem' }}>
        <OverallNumbers startDate={startDate} endDate={endDate} userIds={selectedUsers} />
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="One point per user: valid sessions vs total queries. A valid session is one where the user ran at least 1 query. Click a point to see that user's query list.">
          <UserActivityScatter startDate={startDate} endDate={endDate} userIds={selectedUsers} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Total queries per month across all selected users. Click a point to see the queries from that month.">
          <QueriesOverTime startDate={startDate} endDate={endDate} userIds={selectedUsers} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Per month: new registrations (user accounts created), active users (ran at least 1 query) and sessions (logins). Use the buttons to show or hide each line.">
          <UsageOverTime startDate={startDate} endDate={endDate} userIds={selectedUsers} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Calendar heatmap of daily query activity. Darker days had more queries. Click a day to see its query list.">
          <ActivityCalendar startDate={startDate} endDate={endDate} userIds={selectedUsers} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <QueriesByUserViews startDate={startDate} endDate={endDate} userIds={perUserChartIds} />
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Monthly query counts as one small chart per user, most active first, so each activity shape can be seen on its own without overlapping the others.">
          <QueriesByUserSparklines startDate={startDate} endDate={endDate} userIds={perUserChartIds} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Average complexity score per user, for users with 10 or more queries in this range. Highest first. Click a bar to see that user's queries.">
          <ComplexityByUser startDate={startDate} endDate={endDate} userIds={selectedUsers} />
        </ChartCard>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Failed login attempts by status. SUCCESS is excluded. Click a bar or label to see the list of attempts.">
          <LoginStatus />
        </ChartCard>
      </div>
        </>
      )}
    </div>
  );
}

export default UsersPage;
