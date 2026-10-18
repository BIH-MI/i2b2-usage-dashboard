import { useState } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';
import QueryLinkList from '../QueryLinkList';

const PAGE_SIZE = 10;

const MODES = [
  { key: 'all', label: 'Overall', field: 'concepts' },
  { key: 'included', label: 'Included', field: 'included_concepts' },
  { key: 'excluded', label: 'Excluded', field: 'excluded_concepts' }
];

// counts each concept once per query, even if a query's XML touches it twice.
// toggle chooses whether a concept counts regardless of include/exclude, or only one side
function ConceptFrequency({ queries }) {
  const { colors } = useTheme();
  const [pageIndex, setPageIndex] = useState(0);
  const [mode, setMode] = useState('all');
  const [selectedConcept, setSelectedConcept] = useState(null);

  const field = MODES.find((m) => m.key === mode).field;
  const counts = {};
  const queriesByConcept = {}; // concept -> queries that touch it, a query can appear under several concepts
  queries.forEach((q) => {
    new Set(q[field]).forEach((concept) => {
      counts[concept] = (counts[concept] || 0) + 1;
      queriesByConcept[concept] = queriesByConcept[concept] || [];
      queriesByConcept[concept].push(q);
    });
  });

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePageIndex = Math.min(pageIndex, totalPages - 1); // clamps after the filter/date range changes the list size

  const page = sorted
    .slice(safePageIndex * PAGE_SIZE, safePageIndex * PAGE_SIZE + PAGE_SIZE)
    .reverse(); // reversed so the horizontal bar chart reads highest-at-top

  const option = {
    title: { text: `Most Common Concepts (${MODES.find((m) => m.key === mode).label})`, textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'axis',
      // shows the full path, not just the truncated axis label
      formatter: (params) => {
        const [concept, count] = page[params[0].dataIndex];
        const { prefix, label } = splitConceptPath(concept);
        return `${prefix ? `${prefix} › ` : ''}${label}<br/>Queries: ${count}`;
      },
      extraCssText: 'max-width: 260px; white-space: normal;' // wraps long concept paths instead of one long line
    },
    axisPointer: { type: 'shadow' }, // highlights the whole row on hover, including near the label
    grid: { left: 160 }, // room for the concept labels
    xAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'category',
      data: page.map(([concept]) => splitConceptPath(concept).label),
      axisLabel: { color: colors.textOnPage, width: 150, overflow: 'truncate' },
      triggerEvent: true // lets hovering the label itself (not just the bar) fire the tooltip too
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 20,
        data: page.map(([, count]) => count),
        itemStyle: { color: colors.categorical[0] }
      }
    ]
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.6rem' }}>
        {MODES.map((m) => (
          <button
            key={m.key}
            onClick={() => {
              setMode(m.key);
              setPageIndex(0);
              setSelectedConcept(null);
            }}
            style={{
              padding: '0.3rem 0.7rem',
              borderRadius: '999px',
              border: `1px solid ${colors.tileBorder}`,
              backgroundColor: mode === m.key ? colors.tileBorder : 'transparent',
              color: mode === m.key ? '#fff' : colors.textOnPage,
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {page.length === 0 ? (
        <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No concepts found in this range.</p>
      ) : (
        <Chart
          option={option}
          style={{ height: '320px', width: '100%' }}
          onEvents={{ click: (params) => setSelectedConcept(page[params.dataIndex][0]) }}
        />
      )}

      {selectedConcept && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={queriesByConcept[selectedConcept]} title={splitConceptPath(selectedConcept).label} />
        </div>
      )}

      {page.length > 0 && totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.4rem' }}>
          <button onClick={() => setPageIndex((p) => p - 1)} disabled={safePageIndex === 0}>
            Prev
          </button>
          <span style={{ color: colors.textOnPage, fontSize: '0.8rem' }}>
            {safePageIndex + 1} / {totalPages}
          </span>
          <button onClick={() => setPageIndex((p) => p + 1)} disabled={safePageIndex >= totalPages - 1}>
            Next
          </button>
        </div>
      )}
    </div>
  );
}

export default ConceptFrequency;
