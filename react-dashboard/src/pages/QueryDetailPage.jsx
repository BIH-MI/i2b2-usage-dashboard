import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { useTheme } from '../theme';
import { useUserFilter } from '../context/UserFilterContext';
import { formatDateTime } from '../utils/formatDate';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';
import { statusColorMap } from '../utils/statusColors';
import { constraintRows, definitionKey } from '../utils/queryDefinition';
import ChartCard from '../components/ChartCard';
import LoadingState from '../components/LoadingState';
import QueryDurationHistogram from '../components/charts/QueryDurationHistogram';
import OntologyTree from '../components/charts/OntologyTree';
import Pill from '../components/queryDetail/Pill';
import TimingSection from '../components/queryDetail/TimingSection';
import ConceptsUsedList from '../components/queryDetail/ConceptsUsedList';
import ConstraintsSection from '../components/queryDetail/ConstraintsSection';
import StructureSection from '../components/queryDetail/StructureSection';

// reached from a table row or chart point. Works standalone via query_master_id,
// so it also works opened directly from a URL or refresh
function QueryDetailPage() {
  const { colors, dateFormat } = useTheme();
  const { displayName } = useUserFilter();
  const { queryId } = useParams();
  const [query, setQuery] = useState(null); // null while loading
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState(false);
  const [allQueries, setAllQueries] = useState(null); // every query, for the "compared to others" numbers
  const [allQueriesError, setAllQueriesError] = useState(false);

  useEffect(() => {
    setQuery(null);
    setNotFound(false);
    setError(false);
    fetchWithTimeout(`/api/queries?query_master_id=${encodeURIComponent(queryId)}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => {
        if (rows.length === 0) {
          setNotFound(true);
        } else {
          setQuery(rows[0]); // one row per run of the query, the first one is shown
        }
      })
      .catch(() => setError(true));
  }, [queryId]);

  // loaded once, the comparison doesn't depend on which query is open
  useEffect(() => {
    fetchWithTimeout('/api/queries')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setAllQueries(rows))
      .catch(() => setAllQueriesError(true));
  }, []);

  const backLink = (
    <Link
      to="/queries"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.4rem',
        color: colors.textOnPage,
        fontSize: '0.85rem',
        textDecoration: 'none',
        marginBottom: '1.25rem'
      }}
    >
      <ArrowLeft size={16} /> Back to Queries
    </Link>
  );

  if (error) {
    return (
      <div style={{ padding: '1.5rem 2rem' }}>
        {backLink}
        <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>
      </div>
    );
  }

  if (notFound) {
    return (
      <div style={{ padding: '1.5rem 2rem' }}>
        {backLink}
        <p style={{ color: colors.textOnPage }}>No query found with id {queryId}.</p>
      </div>
    );
  }

  if (!query) {
    return (
      <div style={{ padding: '1.5rem 2rem' }}>
        {backLink}
        <LoadingState height="200px" />
      </div>
    );
  }

  const statusColor = statusColorMap(colors)[query.status_name] || colors.status.neutral;

  // other runs of exactly the same query definition, oldest first
  const sameRuns = allQueries
    ? allQueries
        .filter((q) => q.query_master_id !== query.query_master_id && definitionKey(q) === definitionKey(query))
        .sort((a, b) => new Date(a.start_date) - new Date(b.start_date))
    : [];
  // one row per query, /api/queries returns one row per run of a query
  const uniqueQueries = allQueries ? [...new Map(allQueries.map((q) => [q.query_master_id, q])).values()] : null;

  const rows = constraintRows(query);
  const showHistogram = allQueries && query.duration !== null;

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <div style={{ maxWidth: '1000px' }}>
        {backLink}

        <h2 style={{ color: colors.textOnPage, marginBottom: '0.4rem' }}>{query.name}</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <Pill color={statusColor} filled>{query.status_name}</Pill>
          <span style={{ color: `${colors.textOnPage}99`, fontSize: '0.85rem' }}>
            #{query.query_master_id} · by <strong style={{ color: colors.textOnPage }}>{displayName(query.user_id)}</strong> ·{' '}
            {formatDateTime(query.start_date, dateFormat)}
          </span>
        </div>
      </div>

      {/* rows of two cards (3:2) with equal height. A card without a partner takes the full row, one column on narrow screens */}
      <style>{`
        .query-detail-row {
          display: grid;
          grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }
        .query-detail-row.single {
          grid-template-columns: minmax(0, 1fr);
        }
        /* the concept list fills the height of the timing card and scrolls, so it never makes the row taller */
        .query-detail-fill {
          position: relative;
          min-height: 200px;
        }
        .query-detail-fill > div {
          position: absolute;
          inset: 0;
          overflow-y: auto;
        }
        @media (max-width: 1100px) {
          .query-detail-row {
            grid-template-columns: minmax(0, 1fr);
          }
          .query-detail-fill > div {
            position: static;
            max-height: 420px;
          }
        }
      `}</style>

      {/* row 1: timing and the concepts the query used */}
      <div className="query-detail-row">
        <TimingSection query={query} allQueries={allQueries} allQueriesError={allQueriesError} sameRuns={sameRuns} />
        <div className="query-detail-fill">
          <ConceptsUsedList concepts={query.concepts} queries={uniqueQueries} queriesError={allQueriesError} />
        </div>
      </div>

      {/* row 2: constraints and how the duration compares to all queries */}
      <div className={`query-detail-row${showHistogram ? '' : ' single'}`}>
        <ConstraintsSection query={query} rows={rows} />
        {showHistogram && (
          <ChartCard info="How long all queries took, grouped on a log scale. The orange bar is the group this query falls into.">
            <QueryDurationHistogram queries={allQueries} duration={query.duration} />
          </ChartCard>
        )}
      </div>

      <StructureSection panels={query.panels} />

      {/* ontology tree, full width so its branches have room */}
      <div style={{ marginTop: '1.5rem' }}>
        <ChartCard info="Where the concepts of this query sit in the ontology. Starts at the ontologies (e.g. Diagnose, Labor), click a node to open or close its branch. The number is how many of the query's concepts are below it. Scroll to zoom, drag to move.">
          <OntologyTree concepts={query.concepts} />
        </ChartCard>
      </div>
    </div>
  );
}

export default QueryDetailPage;
