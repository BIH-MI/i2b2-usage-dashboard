import { useState } from 'react';
import { useTheme } from '../../theme';
import ChartCard from '../ChartCard';
import QueriesByUser from './QueriesByUser';
import QueriesByUserStacked from './QueriesByUserStacked';
import QueriesByUserOverlay from './QueriesByUserOverlay';

const VIEWS = [
  {
    key: 'lines',
    label: 'Lines',
    Component: QueriesByUser,
    info: "Query activity per month, one line per user, most active first. Click a point to see that user's queries for that month."
  },
  {
    key: 'stacked',
    label: 'Stacked',
    Component: QueriesByUserStacked,
    info: "Monthly query counts per user, stacked to show total combined volume over time. Click a point to see that user's queries for that month."
  },
  {
    key: 'overlaid',
    label: 'Overlaid',
    Component: QueriesByUserOverlay,
    info: "Monthly query counts per user, overlaid as semi-transparent areas so peaks and shapes can be compared directly. Click a point to see that user's queries for that month."
  }
];

// the three per-user monthly charts in one card, the toggle picks which one is shown
function QueriesByUserViews({ startDate, endDate, userIds }) {
  const { colors } = useTheme();
  const [view, setView] = useState('lines');
  const [pageSize, setPageSize] = useState(5); // kept here so "Show N" stays the same when switching views
  const current = VIEWS.find((v) => v.key === view);
  const CurrentChart = current.Component;

  return (
    <ChartCard info={current.info}>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.6rem' }}>
        {VIEWS.map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            style={{
              padding: '0.3rem 0.7rem',
              borderRadius: '999px',
              border: `1px solid ${colors.tileBorder}`,
              backgroundColor: view === v.key ? colors.tileBorder : 'transparent',
              color: view === v.key ? '#fff' : colors.textOnPage,
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      <CurrentChart
        startDate={startDate}
        endDate={endDate}
        userIds={userIds}
        pageSize={pageSize}
        onPageSizeChange={setPageSize}
      />
    </ChartCard>
  );
}

export default QueriesByUserViews;
