import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import LoadingState from '../LoadingState';

const EMPTY_USER_IDS = [];

const MODES = [
  { key: 'new_users', label: 'New registrations' },
  { key: 'active_users', label: 'Active users' },
  { key: 'sessions', label: 'Sessions' }
];

// per month: new registrations, users with at least 1 query, sessions
function UsageOverTime({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const [months, setMonths] = useState(null); // [{ month: "2025-08", new_users, active_users, sessions }, ...]
  const [selected, setSelected] = useState(MODES.map((m) => m.key));
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return;
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/usage-over-time?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setMonths(rows))
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  // at least one line stays selected
  function toggle(key) {
    setSelected((prev) => {
      if (!prev.includes(key)) return [...prev, key];
      return prev.length > 1 ? prev.filter((k) => k !== key) : prev;
    });
  }

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!months) {
    return <LoadingState height="350px" />;
  }

  const shown = MODES.filter((m) => selected.includes(m.key));

  const option = {
    title: { text: 'Usage Over Time', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: months.map((m) => m.month),
      axisLabel: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: colors.textOnPage }
    },
    series: shown.map((m) => ({
      name: m.label,
      type: 'line',
      smooth: true,
      data: months.map((row) => row[m.key]),
      itemStyle: { color: colors.categorical[MODES.indexOf(m)] }
    }))
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.6rem' }}>
        {MODES.map((m, i) => {
          const isOn = selected.includes(m.key);
          return (
            <button
              key={m.key}
              onClick={() => toggle(m.key)}
              style={{
                padding: '0.3rem 0.7rem',
                borderRadius: '999px',
                border: `1px solid ${colors.categorical[i]}`,
                backgroundColor: isOn ? colors.categorical[i] : 'transparent',
                color: isOn ? '#fff' : colors.textOnPage,
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {m.label}
            </button>
          );
        })}
      </div>

      <Chart option={option} notMerge style={{ height: '350px', width: '100%' }} />
    </div>
  );
}

export default UsageOverTime;
