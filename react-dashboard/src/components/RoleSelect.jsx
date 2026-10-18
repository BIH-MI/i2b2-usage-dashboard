import { useEffect, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import { useTheme } from '../theme';

const ROLE_OPTIONS = [
  { value: 'user', label: 'User' },
  { value: 'admin', label: 'Admin' }
];

// role dropdown in the dashboard look, replaces the plain browser <select>.
// variant "pill" is the small badge in the user list, "field" is the full-width form field
function RoleSelect({ value, onChange, variant = 'field', title }) {
  const { colors } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const isPill = variant === 'pill';
  const isAdmin = value === 'admin';
  const current = ROLE_OPTIONS.find((o) => o.value === value) ?? ROLE_OPTIONS[0];

  // Escape closes the list
  useEffect(() => {
    if (!isOpen) return;
    function onKeyDown(e) {
      if (e.key === 'Escape') setIsOpen(false);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen]);

  const triggerStyle = isPill
    ? {
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.25rem',
        padding: '0.15rem 0.45rem 0.15rem 0.6rem',
        borderRadius: '999px',
        border: `1px solid ${isAdmin ? 'transparent' : colors.cardBorder}`,
        background: isAdmin ? `${colors.tileBorder}1a` : 'transparent',
        color: isAdmin ? colors.tileBorder : colors.textOnPage,
        fontSize: '0.72rem',
        fontWeight: 600,
        boxShadow: 'none'
      }
    : {
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.55rem 0.75rem',
        borderRadius: '10px',
        border: `1px solid ${isOpen ? colors.tileBorder : 'var(--border)'}`,
        background: 'var(--control-bg)',
        color: 'var(--text-h)',
        fontSize: '0.9rem',
        boxShadow: isOpen ? `0 0 0 3px ${colors.tileBorder}22` : 'none'
      };

  return (
    <span style={{ position: 'relative', display: isPill ? 'inline-block' : 'block', flexShrink: 0 }}>
      {/* hover can't be set in inline styles */}
      <style>{`
        .role-option { transition: background 0.15s; }
        .role-option:hover { background: ${colors.tileBorder}14; }
      `}</style>

      <button
        type="button"
        title={title}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((v) => !v)}
        style={triggerStyle}
      >
        {current.label}
        <ChevronDown
          size={isPill ? 13 : 16}
          style={{ transition: 'transform 0.15s', transform: isOpen ? 'rotate(180deg)' : 'none' }}
        />
      </button>

      {isOpen && (
        <>
          {/* invisible layer, a click anywhere outside closes the list */}
          <div onClick={() => setIsOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 60 }} />

          <div
            role="listbox"
            style={{
              position: 'absolute',
              top: 'calc(100% + 0.35rem)',
              [isPill ? 'right' : 'left']: 0,
              width: isPill ? '130px' : '100%',
              boxSizing: 'border-box',
              padding: '0.3rem',
              backgroundColor: colors.chartSurface,
              border: `1px solid ${colors.cardBorder}`,
              borderRadius: '12px',
              boxShadow: '0 16px 40px rgba(0,0,0,0.24), 0 2px 8px rgba(0,0,0,0.12)',
              zIndex: 61
            }}
          >
            {ROLE_OPTIONS.map((option) => {
              const isSelected = option.value === value;
              return (
                <div
                  key={option.value}
                  role="option"
                  aria-selected={isSelected}
                  className="role-option"
                  onClick={() => {
                    setIsOpen(false);
                    if (!isSelected) onChange(option.value);
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.45rem 0.55rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    textAlign: 'left'
                  }}
                >
                  <span style={{ flex: 1, minWidth: 0, color: 'var(--text-h)', fontSize: '0.85rem', fontWeight: isSelected ? 600 : 500 }}>
                    {option.label}
                  </span>
                  {isSelected && <Check size={15} color={colors.tileBorder} style={{ flexShrink: 0 }} />}
                </div>
              );
            })}
          </div>
        </>
      )}
    </span>
  );
}

export default RoleSelect;
