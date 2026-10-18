import { useTheme } from '../theme';

// shared loading placeholder, plain text on the page background, no card
function LoadingState({ height = '80px', width = '100%' }) {
  const { colors } = useTheme();
  return (
    <div style={{ height, width, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <style>{`
        @keyframes loading-pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>
      <span style={{ color: colors.textOnPage, fontSize: '0.8rem', animation: 'loading-pulse 1.4s ease-in-out infinite' }}>
        Loading…
      </span>
    </div>
  );
}

export default LoadingState;
