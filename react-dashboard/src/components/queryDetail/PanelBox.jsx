import { useState } from 'react';
import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';
import { formatDay, formatValue, isReversedRange } from '../../utils/queryDefinition';
import Pill from './Pill';
import MoreToggle from './MoreToggle';

const PREVIEW = 3; // concepts shown before "+N more"

// one panel box: its concepts joined by OR, badges for everything that restricts it.
// Long panels (e.g. 80 concepts) only show the first few until opened
function PanelBox({ panel, index }) {
  const { colors } = useTheme();
  const [open, setOpen] = useState(false);
  const accent = panel.excluded ? colors.status.critical : colors.tileBorder;
  const shownItems = open ? panel.items : panel.items.slice(0, PREVIEW);

  return (
    <div
      style={{
        minWidth: 0, // lets a box wider than the screen shrink, the concept name then gets "…"
        flex: 1, // every panel fills the row, so all right edges line up
        background: `${accent}0d`, // soft tint of the panel color instead of a heavy border
        border: `1px solid ${accent}33`,
        borderRadius: '12px',
        overflow: 'hidden'
      }}
    >
      {/* header strip: panel number and the badges that apply to the whole panel */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          flexWrap: 'wrap',
          padding: '0.5rem 0.9rem',
          background: `${accent}14`,
          borderBottom: `1px solid ${accent}26`
        }}
      >
        <span style={{ color: accent, fontWeight: 700, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Panel {index + 1}
        </span>
        {panel.excluded && <Pill color={colors.status.critical} filled>NOT</Pill>}
        {panel.occurrences > 1 && <Pill color={colors.tileBorder}>at least {panel.occurrences}×</Pill>}
      </div>

      <div style={{ padding: '0.6rem 0.9rem 0.75rem' }}>
        {shownItems.map((item, i) => {
          const { prefix, label } = splitConceptPath(item.key);
          return (
            <div key={i}>
              {/* OR as a small label on a thin line between concepts */}
              {i > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0.45rem 0' }}>
                  <span style={{ flex: 1, height: '1px', background: `${accent}26` }} />
                  <span style={{ color: `${colors.textOnPage}99`, fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.08em' }}>OR</span>
                  <span style={{ flex: 1, height: '1px', background: `${accent}26` }} />
                </div>
              )}
              {/* name first, the ontology path small and muted under it */}
              <div title={item.key} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-h)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
                {prefix && (
                  <div style={{ fontSize: '0.7rem', color: `${colors.textOnPage}99`, overflow: 'hidden', textOverflow: 'ellipsis' }}>{prefix}</div>
                )}
              </div>
              <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap', marginTop: '0.3rem' }}>
                {(item.date_from || item.date_to) && (
                  <Pill color={isReversedRange(item) ? colors.status.serious : colors.textOnPage}>
                    {formatDay(item.date_from)} → {formatDay(item.date_to)}
                  </Pill>
                )}
                {item.value_operator && <Pill color={colors.textOnPage}>{formatValue(item)}</Pill>}
                {/* single-concept panel without badges: invisible badge keeps it as tall as one with a badge */}
                {panel.items.length === 1 && !item.date_from && !item.date_to && !item.value_operator && (
                  <span style={{ visibility: 'hidden' }}>
                    <Pill color={colors.textOnPage}>-</Pill>
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {panel.items.length > PREVIEW && (
          <MoreToggle open={open} hiddenCount={panel.items.length - PREVIEW} onClick={() => setOpen((v) => !v)} />
        )}
      </div>
    </div>
  );
}

export default PanelBox;
