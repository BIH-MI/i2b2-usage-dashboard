import { useTheme } from '../theme';

// shown only in the demo build (VITE_DEMO=true), so nobody takes the invented numbers for real usage
function DemoBanner() {
  const { colors } = useTheme();
  if (import.meta.env.VITE_DEMO !== 'true') return null;

  return (
    <div
      style={{
        padding: '0.4rem 2rem',
        fontSize: '0.8rem',
        color: colors.textOnPage,
        background: `${colors.categorical[3]}26`,
        borderBottom: `1px solid ${colors.categorical[3]}66`
      }}
    >
      Demo: all users, queries and sessions are invented and only show what the dashboard can do.
    </div>
  );
}

export default DemoBanner;
