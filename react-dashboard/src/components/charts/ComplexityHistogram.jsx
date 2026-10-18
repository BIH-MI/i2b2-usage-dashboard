import { useState } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import QueryLinkList from '../QueryLinkList';

const MAX_BARS = 30; // above this, values are grouped into ranges

const MODES = [
  { key: 'score', label: 'Complexity score', value: (q) => q.complexity_score },
  { key: 'concepts', label: 'Concept touch', value: (q) => q.concepts.length },
  { key: 'constraints', label: 'Constraints & temporal', value: (q) => q.num_constraints + q.temporal_query_constraints }
];

// how many queries have each value
function ComplexityHistogram({ queries }) {
  const { colors } = useTheme();
  const [mode, setMode] = useState('score');
  const [selectedBin, setSelectedBin] = useState(null); // label of the clicked bar

  const { label, value } = MODES.find((m) => m.key === mode);
  const unique = [...new Map(queries.map((q) => [q.query_master_id, q])).values()];

  const max = Math.max(0, ...unique.map(value));
  const binSize = max <= MAX_BARS ? 1 : Math.ceil((max + 1) / MAX_BARS);
  const bins = Array.from({ length: Math.floor(max / binSize) + 1 }, (_, i) => {
    const from = i * binSize;
    const to = from + binSize - 1;
    return { label: binSize === 1 ? `${from}` : `${from}–${to}`, queries: [] };
  });
  unique.forEach((q) => bins[Math.floor(value(q) / binSize)].queries.push(q));

  const option = {
    title: { text: `Query Complexity (${label})`, textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      name: label,
      nameLocation: 'middle',
      nameGap: 30,
      data: bins.map((b) => b.label),
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      minInterval: 1,
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'bar',
        data: bins.map((b) => b.queries.length),
        itemStyle: { color: colors.categorical[MODES.findIndex((m) => m.key === mode)] }
      }
    ]
  };

  const clicked = bins.find((b) => b.label === selectedBin);

  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.6rem' }}>
        {MODES.map((m) => (
          <button
            key={m.key}
            onClick={() => {
              setMode(m.key);
              setSelectedBin(null);
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

      {unique.length === 0 ? (
        <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No queries in this range.</p>
      ) : (
        <Chart
          option={option}
          notMerge
          style={{ height: '340px', width: '100%' }}
          onEvents={{ click: (params) => setSelectedBin(params.name) }}
        />
      )}

      {clicked && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={clicked.queries} title={`${label}: ${clicked.label}`} />
        </div>
      )}
    </div>
  );
}

export default ComplexityHistogram;
