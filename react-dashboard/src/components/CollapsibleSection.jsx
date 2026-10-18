import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useTheme } from '../theme';

// wraps a chart section behind a click-to-expand header, for content that can get very tall
function CollapsibleSection({ title, subtitle, defaultOpen = false, children }) {
  const { colors } = useTheme();
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div>
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          cursor: 'pointer',
          userSelect: 'none'
        }}
      >
        {isOpen ? (
          <ChevronDown size={18} color={colors.textOnPage} style={{ flexShrink: 0 }} />
        ) : (
          <ChevronRight size={18} color={colors.textOnPage} style={{ flexShrink: 0 }} />
        )}
        <div>
          <h3 style={{ color: colors.textOnPage, margin: 0 }}>{title}</h3>
          {subtitle && (
            <p style={{ color: `${colors.textOnPage}99`, fontSize: '0.78rem', margin: '0.15rem 0 0' }}>{subtitle}</p>
          )}
        </div>
      </div>

      {isOpen && <div style={{ marginTop: '0.85rem' }}>{children}</div>}
    </div>
  );
}

export default CollapsibleSection;
