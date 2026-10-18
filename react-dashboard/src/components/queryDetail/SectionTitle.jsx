import { useTheme } from '../../theme';

// title at the top of a card,  "Timing"
function SectionTitle({ children }) {
  const { colors } = useTheme();
  return <h3 style={{ color: colors.textOnPage, fontSize: '1rem', margin: '0 0 1rem' }}>{children}</h3>;
}

export default SectionTitle;
