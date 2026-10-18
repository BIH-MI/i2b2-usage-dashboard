import { useState, useEffect } from 'react';
import DateRangeFilter from '../components/DateRangeFilter';
import ChartCard from '../components/ChartCard';
import PageTitle from '../components/PageTitle';
import CollapsibleSection from '../components/CollapsibleSection';
import LoadingState from '../components/LoadingState';
import QueryDurationScatter from '../components/charts/QueryDurationScatter';
import ConceptFrequency from '../components/charts/ConceptFrequency';
import DurationHistogram from '../components/charts/DurationHistogram';
import ConstraintsScatter from '../components/charts/ConstraintsScatter';
import ResultTypesBreakdown from '../components/charts/ResultTypesBreakdown';
import InclusionExclusionSplit from '../components/charts/InclusionExclusionSplit';
import DateConstraintCoverage from '../components/charts/DateConstraintCoverage';
import QueryStatus from '../components/charts/QueryStatus';
import SlowQueriesList from '../components/charts/SlowQueriesList';
import ComplexityHistogram from '../components/charts/ComplexityHistogram';
import { useDateRange } from '../context/DateRangeContext';
import { useUserFilter } from '../context/UserFilterContext';
import { buildDateRangeParams } from '../utils/dateRangeParams';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';
import { useTheme } from '../theme';

function QueriesPage() {
  const { colors } = useTheme();
  const { startDate, endDate, dataRange, error: dateRangeError, setStartDate, setEndDate, applyShortcut, showAllTime } = useDateRange(); //error true if /api/date-range failed
  const { selectedUsers } = useUserFilter();
  const [queries, setQueries] = useState(null);
  const [error, setError] = useState(false); //second error for api/queries 

  useEffect(() => {
    if (!startDate || !endDate) return;
    setQueries(null);
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, selectedUsers);
    fetchWithTimeout(`/api/queries?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setQueries(rows))
      .catch(() => setError(true));
  }, [startDate, endDate, selectedUsers]);

  return (
    <div style={{ padding: '2rem' }}>
      <PageTitle title="Queries" />

      <DateRangeFilter
        startDate={startDate}
        endDate={endDate}
        minDate={dataRange?.min}
        onStartChange={setStartDate}
        onEndChange={setEndDate}
        onApplyShortcut={applyShortcut}
        onAllTime={showAllTime}
      />

      {error || dateRangeError ? ( // /api/queries or /api/date-range failed, show the error message
        <div style={{ marginTop: '1.5rem' }}>
          <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>
        </div>
      ) : !queries ? ( // no errors but queries not loaded yet, show loading, else show charts
        <div style={{ marginTop: '1.5rem' }}>
          <LoadingState height="320px" />
        </div>
      ) : (
        <>
          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 420px', minWidth: 0 }}>
              <ChartCard info="Shows how query duration relates to how many distinct concepts were selected. Click a dot to open that query's detail page.">
                <QueryDurationScatter queries={queries} />
              </ChartCard>
            </div>
            <div style={{ flex: '1 1 420px', minWidth: 0 }}>
              <ChartCard info="Distribution of query durations, bucketed into classes. Only shows queries that took 30 seconds or more. Click a bar to see the list of queries in that duration range.">
                <DurationHistogram queries={queries} />
              </ChartCard>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 420px', minWidth: 0 }}>
              <ChartCard info="Most common concepts used in queries. Toggle buttons switch between all concepts, only included, or only excluded. Click a bar or label to see the list of queries that use that concept.">
                <ConceptFrequency queries={queries} />
              </ChartCard>
            </div>
            <div style={{ flex: '1 1 420px', minWidth: 0 }}>
              <ChartCard info="Percent distribution of how many concepts are included vs excluded across queries. Click a pie slice or label to see the list of queries.">
                <InclusionExclusionSplit queries={queries} />
              </ChartCard>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 420px', minWidth: 0 }}>
              <ChartCard info="How many queries have each complexity. Complexity score = concepts × 1 + constraints × 3 + temporal orderings × 5. Toggle buttons switch to the concept count or the constraint count alone. Every run is counted, also when the same query was run again. Click a bar to see its queries.">
                <ComplexityHistogram queries={queries} />
              </ChartCard>
            </div>
          </div>

          {/* stacked by default, side by side only on wide screens (bigger than a 16" laptop) */}
          <style>{`
            .wide-pair {
              margin-top: 1.5rem;
              display: grid;
              grid-template-columns: 1fr;
              gap: 1.5rem;
            }
            @media (min-width: 1800px) {
              .wide-pair {
                grid-template-columns: 1fr 1fr;
              }
            }
          `}</style>
          <div className="wide-pair">
            <div style={{ minWidth: 0 }}>
              <ChartCard info="Shows how query duration relates to the number of constraints (date, value, occurrence) a query has. Points are colored by whether the query uses temporal ordering. Click a dot to open that query's detail page.">
                <ConstraintsScatter queries={queries} />
              </ChartCard>
            </div>
            <div style={{ minWidth: 0 }}>
              <ChartCard info="Percentage of queries that requested each result type.">
                <ResultTypesBreakdown queries={queries} />
              </ChartCard>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            <ChartCard info="Shows which calendar months are most often requested in query date-range filters, not when the queries themselves were run. Click a bar or scroll to zoom in.">
              <DateConstraintCoverage queries={queries} />
            </ChartCard>
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            <ChartCard info="Number of queries by status. FINISHED is excluded. Click a bar or label to see the list of queries with that status.">
              <QueryStatus queries={queries} />
            </ChartCard>
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            <ChartCard info="The 25 slowest queries in this range. Click a row to expand its constraints, concepts used, and per-result timing.">
              <CollapsibleSection title="Slowest Queries">
                <SlowQueriesList queries={queries} />
              </CollapsibleSection>
            </ChartCard>
          </div>
        </>
      )}
    </div>
  );
}

export default QueriesPage;
