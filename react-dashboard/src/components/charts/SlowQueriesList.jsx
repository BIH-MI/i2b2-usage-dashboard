import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';
import { formatDuration } from '../../utils/formatDuration';

const TOP_N = 25;

// already sorted slowest-first by the backend, renders the top N with a click-to-expand breakdown
function SlowQueriesList({ queries }) {
  const { colors } = useTheme();
  const [expandedId, setExpandedId] = useState(null);

  const top = queries.slice(0, TOP_N);

  return (
    <div>
      {/* same row look as QueryLinkList, hover can't be set in inline styles */}
      <style>{`
        .slow-query-row { transition: background 0.15s; }
        .slow-query-row:hover { background: ${colors.tileBorder}12; }
        .slow-query-name { color: ${colors.textOnPage}; text-decoration: none; }
        .slow-query-name:hover { color: ${colors.tileBorder}; text-decoration: underline; }
      `}</style>

      <div style={{ maxHeight: '480px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.3rem', paddingRight: '0.3rem' }}>
        {top.map((q, i) => {
          const isExpanded = expandedId === i;
          return (
            <div
              key={i}
              style={{
                flexShrink: 0,
                borderRadius: '10px',
                border: `1px solid ${isExpanded ? `${colors.tileBorder}55` : colors.cardBorder}`,
                background: isExpanded ? `${colors.tileBorder}0a` : 'transparent',
                overflow: 'hidden'
              }}
            >
              <div
                className="slow-query-row"
                onClick={() => setExpandedId(isExpanded ? null : i)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.5rem 0.7rem',
                  fontSize: '0.85rem',
                  color: colors.textOnPage,
                  cursor: 'pointer'
                }}
              >
                {isExpanded ? (
                  <ChevronDown size={14} color={colors.tileBorder} style={{ flexShrink: 0 }} />
                ) : (
                  <ChevronRight size={14} color={`${colors.textOnPage}99`} style={{ flexShrink: 0 }} />
                )}
                <span style={{ flexShrink: 0, minWidth: '1.4rem', color: `${colors.textOnPage}99`, fontSize: '0.75rem', fontVariantNumeric: 'tabular-nums' }}>
                  {i + 1}.
                </span>
                {/* same id pill as in QueryLinkList */}
                <span
                  style={{
                    flexShrink: 0,
                    padding: '0.1rem 0.45rem',
                    borderRadius: '999px',
                    background: `${colors.tileBorder}1a`,
                    color: colors.tileBorder,
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    fontVariantNumeric: 'tabular-nums'
                  }}
                >
                  #{q.query_master_id}
                </span>
                <Link
                  to={`/queries/${q.query_master_id}`}
                  onClick={(e) => e.stopPropagation()} // don't also toggle the row's expand/collapse
                  className="slow-query-name"
                  title={q.name}
                  style={{ flex: 1, minWidth: 0, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                >
                  {q.name}
                </Link>
                <span style={{ flexShrink: 0, color: `${colors.textOnPage}99`, fontSize: '0.72rem' }}>{q.status_name}</span>
                {/* duration as a pill, it is what this list is about */}
                <span
                  style={{
                    flexShrink: 0,
                    minWidth: '4rem',
                    textAlign: 'center',
                    padding: '0.1rem 0.5rem',
                    borderRadius: '999px',
                    background: `${colors.status.serious}22`,
                    color: colors.status.serious,
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    fontVariantNumeric: 'tabular-nums'
                  }}
                >
                  {formatDuration(q.duration)}
                </span>
              </div>

              {isExpanded && (
                <div style={{ padding: '0.2rem 0.9rem 0.8rem 2.3rem', fontSize: '0.8rem' }}>
                  <div style={{ color: `${colors.textOnPage}99`, marginBottom: '0.35rem' }}>
                    {q.num_constraints} constraint{q.num_constraints === 1 ? '' : 's'}, {q.temporal_query_constraints}{' '}
                    temporal
                  </div>

                  <div
                    style={{
                      fontSize: '0.68rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      color: `${colors.textOnPage}99`,
                      margin: '0.4rem 0 0.25rem'
                    }}
                  >
                    Concepts used
                  </div>
                  {q.concepts.length === 0 ? (
                    <div style={{ color: colors.textOnPage }}>None</div>
                  ) : (
                    [...new Set(q.concepts)].map((concept, ci) => {
                      const { prefix, label } = splitConceptPath(concept);
                      return (
                        <div key={ci} style={{ color: colors.textOnPage }}>
                          {prefix && <span style={{ color: `${colors.textOnPage}99` }}>{prefix} › </span>}
                          {label}
                        </div>
                      );
                    })
                  )}

                  <div
                    style={{
                      fontSize: '0.68rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      color: `${colors.textOnPage}99`,
                      margin: '0.6rem 0 0.25rem'
                    }}
                  >
                    Per-result timing
                  </div>
                  {q.results.length === 0 ? (
                    <div style={{ color: colors.textOnPage }}>No result breakdown available</div>
                  ) : (
                    q.results.map((r, ri) => (
                      <div key={ri} style={{ display: 'flex', justifyContent: 'space-between', color: colors.textOnPage }}>
                        <span>{r.result_type_name}</span>
                        <span style={{ color: `${colors.textOnPage}99` }}>{formatDuration(r.duration)}</span>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SlowQueriesList;
