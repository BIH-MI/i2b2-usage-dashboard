import { useTheme } from '../../theme';

// small text button used for "+N more" / "Show less"
function MoreToggle({ open, hiddenCount, onClick }) {
  const { colors } = useTheme();
  return (
    <span
      onClick={onClick}
      style={{ display: 'inline-block', marginTop: '0.4rem', fontSize: '0.75rem', fontWeight: 600, color: colors.tileBorder, cursor: 'pointer' }}
    >
      {open ? 'Show less' : `+${hiddenCount} more`}
    </span>
  );
}

export default MoreToggle;
