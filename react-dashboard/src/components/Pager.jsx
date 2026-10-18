import { useState } from 'react';
import { useTheme } from '../theme';

// shared "Show N / Prev / X-Y of Z / Next" control for the paged per-user charts
function Pager({ pageSize, onPageSizeChange, pageIndex, onPageIndexChange, totalItems, pageSizeOptions, label = '' }) {
  const { colors } = useTheme();
  const [draft, setDraft] = useState(String(pageSize)); // what is typed in the box, can be empty while typing
  const [isOpen, setIsOpen] = useState(false); // suggestion list under the box
  const totalPages = Math.ceil(totalItems / pageSize);
  const pageStart = pageIndex * pageSize;

  function applySize(n) {
    setDraft(String(n));
    onPageSizeChange(n);
    onPageIndexChange(0); // page size changed, back to page 1
  }

  return (
    <div>
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem 1rem' }}>
      <label style={{ color: colors.textOnPage, fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
        Show
        <span style={{ position: 'relative', display: 'inline-block' }}>
          <input
            type="text"
            inputMode="numeric"
            value={draft}
            onFocus={(e) => {
              setIsOpen(true);
              e.target.select();
            }}
            onChange={(e) => {
              const digits = e.target.value.replace(/\D/g, ''); // numbers only
              setDraft(digits);
              if (Number(digits) >= 1) applySize(Number(digits));
            }}
            onBlur={() => {
              setIsOpen(false);
              setDraft(String(pageSize)); // empty or 0 goes back to the last valid size
            }}
            onKeyDown={(e) => e.key === 'Enter' && e.target.blur()}
            style={{
              width: '3rem',
              padding: '0.3rem 0.5rem',
              borderRadius: '8px',
              border: `1px solid ${colors.tileBorder}`,
              background: colors.chartSurface,
              color: colors.textOnPage,
              fontSize: '0.85rem',
              textAlign: 'center'
            }}
          />

          {isOpen && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(100% + 0.3rem)',
                left: 0,
                zIndex: 20,
                minWidth: '100%',
                padding: '0.25rem',
                borderRadius: '8px',
                background: colors.chartSurface,
                border: `1px solid ${colors.tileBorder}`,
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.25)'
              }}
            >
              {pageSizeOptions.map((n) => (
                <div
                  key={n}
                  onMouseDown={(e) => {
                    e.preventDefault(); // keeps focus, so onBlur doesn't close the list before the click lands
                    applySize(n);
                    setIsOpen(false);
                  }}
                  style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '6px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    backgroundColor: n === pageSize ? colors.tileBorder + '22' : 'transparent',
                    fontWeight: n === pageSize ? 600 : 400
                  }}
                >
                  {n}
                </div>
              ))}
            </div>
          )}
        </span>
      </label>

      {totalPages > 1 && (
        <>
          <button onClick={() => onPageIndexChange(pageIndex - 1)} disabled={pageIndex === 0}>
            Prev
          </button>
          <span style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>
            {label}
            {pageStart + 1}-{Math.min(pageStart + pageSize, totalItems)} of {totalItems}
          </span>
          <button onClick={() => onPageIndexChange(pageIndex + 1)} disabled={pageIndex >= totalPages - 1}>
            Next
          </button>
        </>
      )}
    </div>

    {/* asked for more than exist, not an error, everyone is shown */}
    {totalItems > 0 && pageSize > totalItems && (
      <p style={{ color: colors.tileBorder, fontSize: '0.78rem', margin: '0.35rem 0 0' }}>
        Note: max number of users is {totalItems}, showing all of them.
      </p>
    )}
    </div>
  );
}

export default Pager;
