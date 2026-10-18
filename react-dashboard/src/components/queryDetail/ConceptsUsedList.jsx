import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';
import CoOccurrenceList from './CoOccurrenceList';
import MoreToggle from './MoreToggle';

const PREVIEW = 10; // rows shown before "+N more"

// same look as "Concepts used" in the query lists. A click on a concept opens
// the concepts most often used together with it. `queries` is null while still loading
function ConceptsUsedList({ concepts, queries, queriesError }) {
  const { colors } = useTheme();
  const [openConcept, setOpenConcept] = useState(null); // concept whose co-occurrences are shown
  const [showAll, setShowAll] = useState(false);
  const used = [...new Set(concepts)];

  return (
    <div
      style={{
        padding: '0.5rem 0.75rem',
        borderLeft: `2px solid ${colors.tileBorder}`,
        background: `${colors.tileBorder}14`,
        boxSizing: 'border-box'
      }}
    >
      <div
        style={{
          fontSize: '0.68rem',
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          color: `${colors.textOnPage}99`,
          marginBottom: '0.4rem'
        }}
      >
        Concepts used ({used.length})
      </div>

      {used.length === 0 ? (
        <span style={{ fontSize: '0.8rem', color: colors.textOnPage }}>No concepts found.</span>
      ) : (
        (showAll ? used : used.slice(0, PREVIEW)).map((concept, i) => {
          const { prefix, label } = splitConceptPath(concept);
          const isOpen = openConcept === concept;
          return (
            <div key={i} style={{ marginBottom: '0.3rem' }}>
              <div
                onClick={() => setOpenConcept(isOpen ? null : concept)}
                title="Show concepts used together with this one"
                style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.82rem', cursor: 'pointer' }}
              >
                {isOpen ? <ChevronDown size={14} color={colors.textOnPage} /> : <ChevronRight size={14} color={colors.textOnPage} />}
                <span>
                  {prefix && <span style={{ color: `${colors.textOnPage}99` }}>{prefix} › </span>}
                  <span style={{ color: colors.textOnPage }}>{label}</span>
                </span>
              </div>
              {isOpen &&
                (queries ? (
                  <CoOccurrenceList concept={concept} queries={queries} path={[concept]} />
                ) : (
                  <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.75rem', marginLeft: '1.2rem' }}>
                    {queriesError ? 'Could not load queries.' : 'Loading…'}
                  </div>
                ))}
            </div>
          );
        })
      )}

      {used.length > PREVIEW && (
        <MoreToggle open={showAll} hiddenCount={used.length - PREVIEW} onClick={() => setShowAll((v) => !v)} />
      )}
    </div>
  );
}

export default ConceptsUsedList;
