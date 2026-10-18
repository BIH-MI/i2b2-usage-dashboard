import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useTheme } from '../theme';
import { formatDateTime } from '../utils/formatDate';
import { splitConceptPath } from '../utils/conceptPath';
import LoadingState from './LoadingState';
import CollapsibleSection from './CollapsibleSection';

const PAGE_SIZE = 20;

// plain list of queries, each one a link to its detail page, expandable to show its concepts.
function QueryLinkList({ queries, title = 'Queries', emptyText = 'No queries found.' }) {
  const { colors, dateFormat } = useTheme();
  const [pageIndex, setPageIndex] = useState(0);
  const [expandedQueryId, setExpandedQueryId] = useState(null); // which row was clicked (query_master_id)

  if (!queries) return <LoadingState height="80px" />;
  if (queries.length === 0) return <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>{emptyText}</p>;

  const totalPages = Math.ceil(queries.length / PAGE_SIZE);
  const safePageIndex = Math.min(pageIndex, totalPages - 1);
  const pageStart = safePageIndex * PAGE_SIZE;
  const page = queries.slice(pageStart, pageStart + PAGE_SIZE);

  return (
    <CollapsibleSection title={`${title} (${queries.length})`} defaultOpen={true}>
      {/* hover and focus can't be set in inline styles */}
      <style>{`
        .query-link-row { transition: background 0.15s; }
        .query-link-row:hover { background: ${colors.tileBorder}12; }
        .query-link-name { color: ${colors.textOnPage}; text-decoration: none; }
        .query-link-name:hover { color: ${colors.tileBorder}; text-decoration: underline; }
      `}</style>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
      {page.map((q) => {
        const isExpanded = expandedQueryId === q.query_master_id;
        return (
          <div
            key={q.query_master_id}
            style={{
              borderRadius: '10px',
              border: `1px solid ${isExpanded ? `${colors.tileBorder}55` : colors.cardBorder}`,
              background: isExpanded ? `${colors.tileBorder}0a` : 'transparent',
              overflow: 'hidden'
            }}
          >
            <div
              className="query-link-row"
              onClick={() => setExpandedQueryId(isExpanded ? null : q.query_master_id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                padding: '0.5rem 0.7rem',
                color: colors.textOnPage,
                fontSize: '0.85rem',
                cursor: 'pointer'
              }}
            >
              {isExpanded ? (
                <ChevronDown size={14} color={colors.tileBorder} style={{ flexShrink: 0 }} />
              ) : (
                <ChevronRight size={14} color={`${colors.textOnPage}99`} style={{ flexShrink: 0 }} />
              )}
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
                onClick={(e) => e.stopPropagation()}
                className="query-link-name"
                title={q.name}
                // long names get "…" instead of pushing the date to a new line
                style={{ flex: 1, minWidth: 0, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              >
                {q.name}
              </Link>
              <span style={{ flexShrink: 0, color: `${colors.textOnPage}99`, fontSize: '0.75rem', fontVariantNumeric: 'tabular-nums' }}>
                {formatDateTime(q.start_date, dateFormat)}
              </span>
            </div>

            {isExpanded && (
              <div style={{ padding: '0.2rem 0.9rem 0.8rem 2.3rem' }}>
                <div
                  style={{
                    fontSize: '0.68rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                    color: `${colors.textOnPage}99`,
                    marginBottom: '0.45rem'
                  }}
                >
                  Concepts used
                </div>

                {q.concepts.length === 0 ? (
                  <span style={{ fontSize: '0.8rem', color: colors.textOnPage }}>No concepts found.</span>
                ) : (
                  [...new Set(q.concepts)].map((concept, ci) => {
                    const { prefix, label } = splitConceptPath(concept);
                    return (
                      <div key={ci} style={{ fontSize: '0.82rem', marginBottom: '0.3rem' }}>
                        {prefix && <span style={{ color: `${colors.textOnPage}99` }}>{prefix} › </span>}
                        <span style={{ color: colors.textOnPage }}>{label}</span>
                      </div>
                    );
                  })
                )}
              </div>
            )}
          </div>
        );
      })}
      </div>

      {totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginTop: '0.75rem' }}>
          <button onClick={() => setPageIndex(safePageIndex - 1)} disabled={safePageIndex === 0}>
            Prev
          </button>
          <span style={{ color: colors.textOnPage, fontSize: '0.8rem' }}>
            {pageStart + 1}-{Math.min(pageStart + PAGE_SIZE, queries.length)} of {queries.length}
          </span>
          <button onClick={() => setPageIndex(safePageIndex + 1)} disabled={safePageIndex >= totalPages - 1}>
            Next
          </button>
        </div>
      )}
    </CollapsibleSection>
  );
}

export default QueryLinkList;
