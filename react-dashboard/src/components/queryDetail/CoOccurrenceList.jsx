import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';

const PAGE_SIZE = 10; // rows shown before "Show all"
const MAX_DEPTH = 3; // how deep the tree can be opened

// counts which concepts appear in the same query as `concept`, most frequent first
function coOccurrences(concept, queries) {
  const counts = {}; // for each other concept: in how many queries it is used together with `concept`
  let withConcept = 0; // queries that contain the concept itself
  queries.forEach((q) => {
    const group = new Set(q.concepts); // each concept once, even if the query uses it in two panels
    if (!group.has(concept)) return;
    withConcept += 1;
    group.forEach((other) => {
      if (other !== concept) counts[other] = (counts[other] || 0) + 1; 
    });
  });
  // e.g. { sorted: [["Diabetes", 2], ["HbA1c", 1]], withConcept: 3 }, the list then shows 2/3 and 1/3 as percentages
  return { sorted: Object.entries(counts).sort((a, b) => b[1] - a[1]), withConcept };
}

// list under a clicked concept, each row can be opened again for the next level.
// `path` holds the concepts already open above, so the tree never goes back to them
function CoOccurrenceList({ concept, queries, path }) {
  const { colors } = useTheme();
  const [openConcept, setOpenConcept] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const { sorted, withConcept } = coOccurrences(concept, queries);
  const rows = sorted.filter(([other]) => !path.includes(other)); // no loops back up the tree
  const shown = showAll ? rows : rows.slice(0, PAGE_SIZE);
  const canOpen = path.length < MAX_DEPTH;
  const queryWord = withConcept === 1 ? 'query' : 'queries';
  const conceptWord = rows.length === 1 ? 'concept' : 'concepts';

  return (
    <div
      style={{
        margin: '0.3rem 0 0.5rem 0.6rem',
        padding: '0.4rem 0.6rem',
        borderLeft: `2px solid ${colors.tileBorder}`,
        background: `${colors.tileBorder}14`
      }}
    >
      <div style={{ fontSize: '0.72rem', color: `${colors.textOnPage}99`, marginBottom: '0.3rem' }}>
        Used in {withConcept} {queryWord}. Used together with {rows.length} {conceptWord}:
      </div>

      {rows.length === 0 && (
        <div style={{ fontSize: '0.8rem', color: colors.textOnPage }}>No other concepts.</div>
      )}

      {shown.map(([other, count]) => {
        const percent = Math.round((count / withConcept) * 100); // share of the queries that use the parent concept
        const isOpen = openConcept === other;
        const { prefix, label } = splitConceptPath(other);
        return (
          <div key={other}>
            <div
              onClick={() => canOpen && setOpenConcept(isOpen ? null : other)}
              title={other}
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.15rem 0', cursor: canOpen ? 'pointer' : 'default', fontSize: '0.8rem' }}
            >
              {canOpen ? (isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />) : <span style={{ width: 14 }} />}
              <span style={{ flex: 1, minWidth: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {prefix && <span style={{ color: `${colors.textOnPage}99` }}>{prefix} › </span>}
                <span style={{ color: colors.textOnPage }}>{label}</span>
              </span>
              {/* small bar, share of the parent's queries */}
              <span style={{ width: '50px', height: '6px', borderRadius: '3px', background: `${colors.tileBorder}33`, flexShrink: 0 }}>
                <span style={{ display: 'block', width: `${percent}%`, height: '100%', borderRadius: '3px', background: colors.tileBorder }} />
              </span>
              <span style={{ color: `${colors.textOnPage}99`, whiteSpace: 'nowrap', minWidth: '4.5rem', textAlign: 'right' }}>
                {count} · {percent}%
              </span>
            </div>
            {isOpen && <CoOccurrenceList concept={other} queries={queries} path={[...path, other]} />}
          </div>
        );
      })}

      {rows.length > PAGE_SIZE && (
        <button onClick={() => setShowAll((v) => !v)} style={{ marginTop: '0.3rem', fontSize: '0.75rem' }}>
          {showAll ? 'Show less' : `Show all ${rows.length} concepts`}
        </button>
      )}
    </div>
  );
}

export default CoOccurrenceList;
