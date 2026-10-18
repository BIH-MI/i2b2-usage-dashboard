import { useTheme } from '../theme';
import InfoTooltip from './InfoTooltip';

// wraps one chart in its own card surface, so the page reads as a grid of panels
function ChartCard({ children, info = '' }) {
  const { colors } = useTheme();

  return (
    <div
      style={{
        position: 'relative',
        height: '100%',
        boxSizing: 'border-box',
        backgroundColor: colors.chartSurface,
        border: `1px solid ${colors.cardBorder}`,
        borderRadius: '14px',
        padding: '1.25rem',
        boxShadow: colors.cardShadow
      }}
    >
      <InfoTooltip text={info} />
      {children}
    </div>
  );
}

export default ChartCard;
