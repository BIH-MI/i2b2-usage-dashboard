import Chart from '../Chart';
import { useEffect, useState } from 'react';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { formatDateTime } from '../../utils/formatDate';
import LoadingState from '../LoadingState';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';

// fixed left-to-right order on the x-axis, matches the severity escalation:
// NONEXIST (typo) < BADPASSWORD (wrong password) < LOCKEDOUT (most severe)
const STATUS_ORDER = ['NONEXIST', 'BADPASSWORD', 'LOCKEDOUT'];

function LoginStatus() {
  const { colors, dateFormat } = useTheme();
  const { displayName } = useUserFilter();
  // status colors, not categorical, severity escalates left to right, matches STATUS_ORDER
  const STATUS_COLOR_MAP = {
    SUCCESS: colors.status.good,
    NONEXIST: colors.status.warning,
    BADPASSWORD: colors.status.serious,
    LOCKEDOUT: colors.status.critical
  };
  const [statusCounts, setStatusCounts] = useState(null); // [{ attempt_cd, count }, ...]
  const [selectedStatus, setSelectedStatus] = useState(null); // which bar was clicked
  const [attempts, setAttempts] = useState(null); // [{ user_id, entry_date }, ...] for selectedStatus
  const [error, setError] = useState(false);

  useEffect(() => {
    fetchWithTimeout('/api/login-status-counts')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setStatusCounts(rows))
      .catch(() => setError(true));
  }, []);

  useEffect(() => {
    if (!selectedStatus) return;
    setAttempts(null);
    fetchWithTimeout(`/api/login-attempts-by-status?attempt_cd=${encodeURIComponent(selectedStatus)}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setAttempts(rows))
      .catch(() => setAttempts([]));
  }, [selectedStatus]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!statusCounts) {
    return <LoadingState height="320px" width="500px" />;
  }

  // SUCCESS excluded, zero-count statuses dropped, sorted by severity not count
  const problemStatusCounts = statusCounts
    .filter((s) => s.attempt_cd !== 'SUCCESS' && s.count > 0)
    .sort((a, b) => STATUS_ORDER.indexOf(a.attempt_cd) - STATUS_ORDER.indexOf(b.attempt_cd));

  const option = {
    title: { text: 'Login Attempts by Status', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: problemStatusCounts.map((s) => s.attempt_cd),
      axisLabel: { color: colors.textOnPage },
      triggerEvent: true // labels fire click events too, same as the bars
    },
    yAxis: {
      type: 'value',
      name: 'Attempts',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 44, // same bar look as "Queries by Status": width, count on top, rounded tops
        label: { show: true, position: 'top', color: colors.textOnPage, fontSize: 11 },
        data: problemStatusCounts.map((s) => ({
          value: s.count,
          itemStyle: { color: STATUS_COLOR_MAP[s.attempt_cd] || colors.status.neutral, borderRadius: [4, 4, 0, 0] }
        }))
      }
    ]
  };

  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1.5rem', width: '100%' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ color: colors.textOnPage, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
          Click a bar or label to see which users had that status.
        </p>
        <Chart
          option={option}
          style={{ height: '320px', width: '100%' }}
          onEvents={{
            click: (params) => {
              // bar click gives params.name, axis label click gives params.value
              setSelectedStatus(params.componentType === 'xAxis' ? params.value : params.name);
            }
          }}
        />
      </div>

      {selectedStatus && (
        <div style={{ flex: 1, minWidth: 0, marginTop: '1.6rem' }}>
          <h4 style={{ color: colors.textOnPage, fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            {selectedStatus} ({attempts ? attempts.length : '…'})
          </h4>

          {!attempts ? (
            <LoadingState height="120px" />
          ) : attempts.length === 0 ? (
            <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No attempts found.</p>
          ) : (
            <div style={{ maxHeight: '320px', overflowY: 'auto' }}>
              {attempts.map((a, i) => (
                <div
                  key={`${a.user_id}-${a.entry_date}-${i}`}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '0.4rem 0.6rem',
                    borderBottom: `1px solid ${colors.tileBorder}22`,
                    fontSize: '0.85rem',
                    color: colors.textOnPage
                  }}
                >
                  <span>{displayName(a.user_id)}</span>
                  <span style={{ color: colors.textSecondary }}>{formatDateTime(a.entry_date, dateFormat)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default LoginStatus;
