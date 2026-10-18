import { Navigate, useLocation } from 'react-router-dom';
import { useCurrentUser } from '../hooks/useCurrentUser';
import LoadingState from './LoadingState';

// shows the pages only after /api/me confirms the login, otherwise sends to the login page
function RequireAuth({ children }) {
  const currentUser = useCurrentUser();
  const location = useLocation();

  if (currentUser === undefined) return <LoadingState height="100svh" />;
  if (currentUser === null) return <Navigate to="/login" replace state={{ from: location }} />;
  if (currentUser === false) {
    // backend down or timeout, the login state is unknown so stay here instead of going to the login page
    return (
      <div style={{ minHeight: '100svh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
        <p>The server is not reachable right now.</p>
        <button type="button" onClick={() => window.location.reload()}>Try again</button>
      </div>
    );
  }
  return children;
}

export default RequireAuth;
