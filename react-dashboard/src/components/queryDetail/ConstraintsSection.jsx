import { useTheme } from '../../theme';
import ChartCard from '../ChartCard';
import SectionTitle from './SectionTitle';
import Tile from './Tile';
import ConceptName from './ConceptName';

// counts per constraint type, then every constraint as one row. `rows` come from constraintRows()
function ConstraintsSection({ query, rows }) {
  const { colors } = useTheme();
  const mutedText = { color: `${colors.textOnPage}99`, fontSize: '0.85rem' };
  const countOf = (type) => rows.filter((r) => r.type === type).length;

  return (
    <ChartCard info="Every restriction set in this query: date ranges, value limits, minimum occurrences and temporal ordering.">
      <SectionTitle>Constraints</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '1rem' }}>
        <Tile label="Total" value={query.num_constraints} />
        <Tile label="Date" value={countOf('Date')} />
        <Tile label="Value" value={countOf('Value')} />
        <Tile label="Occurrence" value={countOf('Occurrence')} />
        <Tile label="Temporal ordering" value={query.temporal_query_constraints > 0 ? 'Yes' : 'No'} />
      </div>

      {rows.length > 0 && (
        // long lists scroll inside a fixed height instead of stretching the card
        <div style={{ marginTop: '1.25rem', maxHeight: '320px', overflowY: 'auto', paddingRight: '0.5rem' }}>
          {rows.map((r, i) => (
            <div
              key={i}
              style={{
                display: 'grid',
                gridTemplateColumns: '90px 70px 1fr auto',
                gap: '0.75rem',
                alignItems: 'baseline',
                padding: '0.4rem 0',
                borderBottom: `1px solid ${colors.tileBorder}22`,
                fontSize: '0.85rem'
              }}
            >
              <span style={{ color: colors.textOnPage, fontWeight: 600 }}>{r.type}</span>
              <span style={mutedText}>Panel {r.panel}</span>
              <span style={{ minWidth: 0 }}>
                {r.concepts.map((c, j) => (
                  <span key={j}>
                    {j > 0 && <span style={mutedText}> OR </span>}
                    <ConceptName concept={c} />
                  </span>
                ))}
              </span>
              <span style={{ color: r.warning ? colors.status.serious : colors.textOnPage, whiteSpace: 'nowrap' }} title={r.warning || undefined}>
                {r.detail}
                {r.warning && ' ⚠'}
              </span>
            </div>
          ))}
        </div>
      )}
    </ChartCard>
  );
}

export default ConstraintsSection;
