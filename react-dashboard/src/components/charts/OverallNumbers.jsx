import { useState, useEffect } from 'react';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import { useTheme } from '../../theme';
import StatTile from '../StatTile';
import LoadingState from '../LoadingState';

// module-level constant, always the same array reference so the useEffect below
// doesn't see a "new" userIds on every render when the caller passes none
const EMPTY_USER_IDS = [];

const TILES = [
  { key: 'query_count', label: 'Number of queries' },
  { key: 'user_count', label: 'Number of users' },
  { key: 'concept_count', label: 'Unique concepts' },
  { key: 'temporal_count', label: 'Temporal queries' },
  {
    key: 'avg_complexity',
    label: 'Avg query complexity',
    info: 'Average complexity score of the queries in this range. Complexity score = concepts × 1 + constraints × 3 + temporal orderings × 5. Constraints are date, value and occurrence limits.'
  },
  { key: 'avg_queries_per_user', label: 'Avg queries per user' }
];

function OverallNumbers({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const [totals, setTotals] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return;
    setTotals(null);
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/overall-stats?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((stats) => setTotals(stats))
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

  if (!totals) {
    return <LoadingState height="95px" />;
  }

  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <style>{`
        .overall-tiles {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 0.75rem;
          max-width: 1140px;
        }
        @media (max-width: 980px) {
          .overall-tiles {
            grid-template-columns: repeat(3, 1fr);
            max-width: 660px;
          }
        }
      `}</style>
      <div className="overall-tiles">
        {TILES.map(({ key, label, info }) => (
          <StatTile key={key} value={totals[key]} label={label} info={info} />
        ))}
      </div>
    </div>
  );
}

export default OverallNumbers;
