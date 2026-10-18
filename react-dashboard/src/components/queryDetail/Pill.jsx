// small rounded badge. `filled` gives it a colored background instead of only a border
function Pill({ children, color, filled = false }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.1rem 0.5rem',
        borderRadius: '999px',
        fontSize: '0.7rem',
        fontWeight: 600,
        border: `1px solid ${color}`,
        color: filled ? '#fff' : color,
        backgroundColor: filled ? color : 'transparent',
        whiteSpace: 'nowrap'
      }}
    >
      {children}
    </span>
  );
}

export default Pill;
