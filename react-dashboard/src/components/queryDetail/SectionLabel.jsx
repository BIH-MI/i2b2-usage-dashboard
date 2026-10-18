import { useTheme } from '../../theme';

// small gray uppercase label above a value, "DURATION"
function SectionLabel({ children }) {
  const { colors } = useTheme();
  return (
    <div
      style={{
        fontSize: '0.7rem',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        color: `${colors.textOnPage}99`,
        marginBottom: '0.4rem'
      }}
    >
      {children}
    </div>
  );
}

export default SectionLabel;
