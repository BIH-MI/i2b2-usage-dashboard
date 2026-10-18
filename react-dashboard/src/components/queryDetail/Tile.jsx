import { useTheme } from '../../theme';
import SectionLabel from './SectionLabel';

// label, one big value and an optional small line under it
function Tile({ label, value, sub }) {
  const { colors } = useTheme();
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div style={{ color: colors.textOnPage, fontSize: '1.1rem', fontWeight: 600 }}>{value}</div>
      {sub && <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.78rem', marginTop: '0.2rem' }}>{sub}</div>}
    </div>
  );
}

export default Tile;
