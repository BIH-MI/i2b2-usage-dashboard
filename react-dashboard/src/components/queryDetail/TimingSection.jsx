import { Link } from 'react-router-dom';
import { useTheme } from '../../theme';
import { useUserFilter } from '../../context/UserFilterContext';
import { formatDateTime } from '../../utils/formatDate';
import { formatDuration } from '../../utils/formatDuration';
import ChartCard from '../ChartCard';
import SpeedGauge from '../charts/SpeedGauge';
import SectionTitle from './SectionTitle';
import SectionLabel from './SectionLabel';
import Tile from './Tile';
import Pill from './Pill';

// how this query's duration compares to every other query that has one
function speedCompared(query, allQueries, allQueriesError, colors) {
  if (allQueriesError) return { value: '-', sub: null, percent: null, color: colors.tileBorder };
  if (!allQueries || query.duration === null) return { value: '…', sub: null, percent: null, color: colors.tileBorder };

  const others = allQueries.filter((q) => q.duration !== null && q.query_master_id !== query.query_master_id);
  const slowerCount = others.filter((q) => q.duration > query.duration).length;
  const fasterCount = others.filter((q) => q.duration < query.duration).length;
  const fasterThan = others.length ? Math.round((slowerCount / others.length) * 100) : 0;
  const slowerThan = others.length ? Math.round((fasterCount / others.length) * 100) : 0;
  const isFaster = fasterThan >= slowerThan; // show whichever side is bigger
  return {
    value: isFaster ? `Faster than ${fasterThan}%` : `Slower than ${slowerThan}%`,
    sub: `of ${others.length} other queries`,
    percent: isFaster ? fasterThan : slowerThan,
    color: isFaster ? colors.status.good : colors.status.serious
  };
}

// duration with the gauge, how often the same query was run, and the time per result type
function TimingSection({ query, allQueries, allQueriesError, sameRuns }) {
  const { colors, dateFormat } = useTheme();
  const { displayName } = useUserFilter();
  const mutedText = { color: `${colors.textOnPage}99`, fontSize: '0.85rem' };
  const speed = speedCompared(query, allQueries, allQueriesError, colors);
  const runUsers = new Set([query.user_id, ...sameRuns.map((q) => q.user_id)]);
  const userWord = runUsers.size === 1 ? 'user' : 'users';

  // text under "Same query run": nothing while loading, then who ran it
  let runsSub = null;
  if (allQueries && sameRuns.length > 0) runsSub = `by ${runUsers.size} ${userWord}`;
  else if (allQueries) runsSub = 'only this run';

  let runsValue = '…';
  if (allQueriesError) runsValue = '-';
  else if (allQueries) runsValue = `${sameRuns.length + 1}×`;

  return (
    <ChartCard info="How long this query took, how that compares to all other queries, and how often exactly this query was run.">
      <SectionTitle>Timing</SectionTitle>
      {/* left: duration with the gauge under it, right: repeat runs with their links */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.5rem' }}>
        <div>
          <Tile label="Duration" value={formatDuration(query.duration)} />
          <div style={{ marginTop: '1rem', maxWidth: '240px' }}>
            <SectionLabel>Compared to others</SectionLabel>
            {speed.percent !== null && <SpeedGauge percent={speed.percent} color={speed.color} />}
            <div style={{ color: speed.color, fontSize: '0.95rem', fontWeight: 600, textAlign: 'center' }}>{speed.value}</div>
            {speed.sub && <div style={{ ...mutedText, fontSize: '0.78rem', textAlign: 'center' }}>{speed.sub}</div>}
          </div>
        </div>

        <div>
          <Tile label="Same query run" value={runsValue} sub={runsSub} />
          {sameRuns.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <SectionLabel>Other runs</SectionLabel>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {sameRuns.map((q) => (
                  <Link
                    key={q.query_master_id}
                    to={`/queries/${q.query_master_id}`}
                    title={`${displayName(q.user_id)} · ${formatDateTime(q.start_date, dateFormat)} · ${formatDuration(q.duration)}`}
                    style={{ textDecoration: 'none' }}
                  >
                    <Pill color={colors.tileBorder}>#{q.query_master_id}</Pill>
                  </Link>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: '1.25rem' }}>
        <SectionLabel>Per-result timing</SectionLabel>
        {query.results.length === 0 ? (
          <div style={mutedText}>No result breakdown available</div>
        ) : (
          query.results.map((r, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '0.3rem 0',
                borderBottom: `1px solid ${colors.tileBorder}22`,
                fontSize: '0.88rem',
                color: colors.textOnPage
              }}
            >
              <span>{r.result_type_name}</span>
              <span style={{ color: `${colors.textOnPage}99` }}>{formatDuration(r.duration)}</span>
            </div>
          ))
        )}
      </div>
    </ChartCard>
  );
}

export default TimingSection;
