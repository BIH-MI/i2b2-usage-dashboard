import { useState } from 'react';
import { Info } from 'lucide-react';
import { useTheme } from '../theme';

// small info icon with a description box.
function InfoTooltip({ text = '', corner = true }) {
  const { colors } = useTheme();
  const [hovered, setHovered] = useState(false);

  return (
    <div
      style={
        corner
          ? { position: 'absolute', top: '0.75rem', right: '0.75rem', zIndex: 20 }
          : { position: 'relative', display: 'inline-flex', zIndex: 20 }
      }
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* sizes come from index.css (--icon-size, --info-width, --info-font), they grow on big screens */}
      <Info
        color={`${colors.textOnPage}99`}
        style={{ cursor: 'help', display: 'block', width: 'var(--icon-size)', height: 'var(--icon-size)' }}
      />

      {hovered && (
        <div
          style={{
            position: 'absolute',
            top: 'calc(var(--icon-size) + 0.45rem)',
            [corner ? 'right' : 'left']: 0, //if true, icon is on the right side , and box opens left
            width: 'var(--info-width)',
            padding: '0.6rem 0.75rem',
            borderRadius: '8px',
            background: colors.chartSurface,
            border: `1px solid ${colors.tileBorder}`,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.25)',
            color: colors.textOnPage,
            fontSize: 'var(--info-font)',
            lineHeight: 1.45
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}

export default InfoTooltip;
