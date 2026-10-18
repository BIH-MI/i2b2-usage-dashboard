import { useTheme } from '../theme';
import InfoTooltip from './InfoTooltip';

// one card, one number + one label. Same surface as ChartCard, so it follows the theme
// info is optional, it adds the info icon in the top right corner
function StatTile({ value, label, info }) {
  const { colors } = useTheme();
  const tileStyle = {
    position: 'relative', // the info icon is placed relative to the tile
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
    boxSizing: 'border-box',
    padding: '0.75rem 0.9rem',
    backgroundColor: colors.chartSurface,
    borderRadius: '14px',
    border: `1px solid ${colors.cardBorder}`,
    boxShadow: colors.cardShadow
  };

  const labelStyle = {
    fontSize: '0.65rem',
    fontWeight: 600,
    color: colors.textOnPage,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    whiteSpace: 'nowrap',
    paddingRight: info ? 'calc(var(--icon-size) + 0.25rem)' : 0 // keeps the label clear of the info icon
  };

  const valueStyle = {
    fontSize: '1.35rem',
    fontWeight: 700,
    color: 'var(--text-h)',
    fontVariantNumeric: 'proportional-nums',
    letterSpacing: '-0.02em'
  };

  return (
    <div style={tileStyle}>
      {info && <InfoTooltip text={info} />}
      <div style={labelStyle}>{label}</div>
      <div style={valueStyle}>{value.toLocaleString()}</div>
    </div>
  );
}

export default StatTile;
