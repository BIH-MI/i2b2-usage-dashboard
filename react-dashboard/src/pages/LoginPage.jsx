import { useState } from 'react';
import { BarChart3, Sun, Moon } from 'lucide-react';
import { useTheme } from '../theme';
import { Navigate, useNavigate, useLocation } from 'react-router-dom';
import { apiUrl } from '../utils/fetchWithTimeout';
import { useCurrentUser } from '../hooks/useCurrentUser';

function LoginPage() {
  const { colors, mode, toggleMode } = useTheme();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();
  const currentUser = useCurrentUser();
  const from = location.state?.from?.pathname || '/users'; // the page the user wanted before being sent here

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');

    let res;
    try {
      res = await fetch(apiUrl('/api/login'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
    } catch {
      setError('Login is not available right now');
      return;
    }

    if (res.ok) {
      navigate(from, { replace: true });
    } else if (res.status === 401 || res.status === 422) { // 422 = input too long or a field missing
      setError('Invalid username or password');
    } else {
      setError('Login is not available right now');
    }
  }


  if (currentUser) return <Navigate to="/users" replace />; // already logged in

  const inputStyle = {
    width: '100%',
    padding: '0.65rem 0.85rem',
    borderRadius: '10px',
    border: '1px solid var(--border)',
    background: 'var(--control-bg)',
    color: 'var(--text-h)',
    fontSize: '0.95rem',
    boxSizing: 'border-box',
    outline: 'none'
  };

  const labelSpanStyle = {
    display: 'block',
    marginBottom: '0.4rem',
    color: colors.textOnPage,
    fontSize: '0.8rem',
    fontWeight: 600
  };

  return (
    <div
      style={{
        position: 'relative',
        minHeight: '100svh',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '1.25rem',
        padding: '1rem',
        boxSizing: 'border-box',
        background: 'transparent' // lets the page gradient from index.css show through
      }}
    >
      {/* same light/dark switch as in the sidebar, also bottom left */}
      <button
        type="button"
        onClick={toggleMode}
        title={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        style={{
          position: 'absolute',
          bottom: '1rem',
          left: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.45rem 0.75rem',
          borderRadius: '10px',
          border: `1px solid ${colors.cardBorder}`,
          background: colors.chartSurface,
          color: colors.textOnPage,
          fontSize: '0.8rem',
          cursor: 'pointer'
        }}
      >
        {mode === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        {mode === 'dark' ? 'Light mode' : 'Dark mode'}
      </button>

      <form
        onSubmit={handleSubmit}
        style={{
          width: '100%',
          maxWidth: '380px',
          boxSizing: 'border-box',
          backgroundColor: colors.chartSurface,
          border: `1px solid ${colors.cardBorder}`,
          borderRadius: '16px',
          padding: '2.25rem 2rem 2rem',
          boxShadow: colors.cardShadow
        }}
      >
        {/* small logo mark in the theme's blue */}
        <div
          style={{
            width: '48px',
            height: '48px',
            margin: '0 auto 1rem',
            borderRadius: '14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `linear-gradient(135deg, ${colors.tileBorder}, ${colors.categorical[6]})`,
            boxShadow: `0 6px 16px ${colors.tileBorder}55`
          }}
        >
          <BarChart3 size={24} color="#fff" />
        </div>

        <h2 style={{ color: 'var(--text-h)', textAlign: 'center', margin: '0 0 0.3rem' }}>i2b2 Usage Dashboard</h2>
        <p style={{ color: colors.textOnPage, textAlign: 'center', fontSize: '0.85rem', marginBottom: '1.75rem' }}>
          Sign in with your i2b2 account
        </p>

        <label style={{ display: 'block', marginBottom: '1rem' }}>
          <span style={labelSpanStyle}>Username</span>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} style={inputStyle} />
        </label>

        <label style={{ display: 'block', marginBottom: '1.5rem' }}>
          <span style={labelSpanStyle}>Password</span>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} style={inputStyle} />
        </label>
        {error && (
          <p style={{ color: '#d64545', fontSize: '0.85rem', margin: '0 0 1rem' }}>
            {error}
          </p>
        )}

        <button
          type="submit"
          style={{
            width: '100%',
            padding: '0.7rem',
            borderRadius: '10px',
            border: 'none',
            background: colors.tileBorder,
            color: '#fff',
            fontSize: '0.95rem',
            fontWeight: 600,
            cursor: 'pointer',
            boxShadow: `0 4px 12px ${colors.tileBorder}44`
          }}
        >
          Log in
        </button>
      </form>
    </div>
  );
}

export default LoginPage;
