import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTheme } from '../theme';
import { apiUrl } from '../utils/fetchWithTimeout';

// same limits as PasswordChange in backend/Endpoints/models.py
const MIN_LENGTH = 8;
const MAX_LENGTH = 64;

// the logged in user changes their own password, same backdrop and card look as the logout dialog
function ChangePasswordDialog({ onClose }) {
  const { colors } = useTheme();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [repeatPassword, setRepeatPassword] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isDone, setIsDone] = useState(false);

  // Escape closes it, like clicking the backdrop
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');

    // checked here first so the user does not wait for the backend for simple mistakes
    if (newPassword.length < MIN_LENGTH || newPassword.length > MAX_LENGTH) {
      setError(`The new password needs ${MIN_LENGTH} to ${MAX_LENGTH} characters.`);
      return;
    }
    if (newPassword !== repeatPassword) {
      setError('The new passwords do not match.');
      return;
    }

    setIsSaving(true);
    let res;
    try {
      res = await fetch(apiUrl('/api/change-password'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
      });
    } catch {
      setError('The server is not reachable right now.');
      setIsSaving(false);
      return;
    }
    setIsSaving(false);

    if (res.ok) {
      setIsDone(true);
    } else if (res.status === 401) {
      window.location.assign(import.meta.env.BASE_URL + 'login'); // session expired
    } else if (res.status === 400) {
      const body = await res.json().catch(() => null);
      setError(body?.detail || 'The password could not be changed.'); // e.g. current password is wrong
    } else if (res.status === 422) {
      setError(`The new password needs ${MIN_LENGTH} to ${MAX_LENGTH} characters.`);
    } else {
      setError('The password could not be changed. Try again later.');
    }
  }

  const inputStyle = {
    width: '100%',
    padding: '0.55rem 0.75rem',
    borderRadius: '10px',
    border: '1px solid var(--border)',
    background: 'var(--control-bg)',
    color: 'var(--text-h)',
    fontSize: '0.9rem',
    boxSizing: 'border-box',
    outline: 'none'
  };

  const labelSpanStyle = {
    display: 'block',
    marginBottom: '0.3rem',
    color: colors.textOnPage,
    fontSize: '0.78rem',
    fontWeight: 600
  };

  const buttonBase = {
    padding: '0.55rem 1rem',
    borderRadius: '10px',
    fontSize: '0.9rem',
    fontWeight: 600,
    cursor: 'pointer'
  };

  return createPortal(
    <>
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.35)', zIndex: 999 }}
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="change-password-title"
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '360px',
          maxWidth: 'calc(100vw - 2rem)',
          boxSizing: 'border-box',
          padding: '1.5rem',
          backgroundColor: colors.chartSurface,
          border: `1px solid ${colors.cardBorder}`,
          borderRadius: '16px',
          boxShadow: '0 16px 40px rgba(0,0,0,0.24), 0 2px 8px rgba(0,0,0,0.12)',
          zIndex: 1000
        }}
      >
        <h3 id="change-password-title" style={{ color: 'var(--text-h)', margin: '0 0 1rem', fontSize: '1.05rem' }}>
          Change password
        </h3>

        {isDone ? (
          <>
            <p style={{ color: colors.textOnPage, fontSize: '0.88rem', margin: '0 0 1.25rem' }}>
              Your password was changed. Use the new one the next time you sign in.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={onClose}
                autoFocus
                style={{ ...buttonBase, border: 'none', background: colors.tileBorder, color: '#fff' }}
              >
                Close
              </button>
            </div>
          </>
        ) : (
          <form onSubmit={handleSubmit}>
            <label style={{ display: 'block', marginBottom: '0.8rem' }}>
              <span style={labelSpanStyle}>Current password</span>
              <input
                type="password"
                autoComplete="current-password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                autoFocus
                style={inputStyle}
              />
            </label>

            <label style={{ display: 'block', marginBottom: '0.8rem' }}>
              <span style={labelSpanStyle}>New password (at least {MIN_LENGTH} characters)</span>
              <input
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                style={inputStyle}
              />
            </label>

            <label style={{ display: 'block', marginBottom: '1rem' }}>
              <span style={labelSpanStyle}>Repeat new password</span>
              <input
                type="password"
                autoComplete="new-password"
                value={repeatPassword}
                onChange={(e) => setRepeatPassword(e.target.value)}
                style={inputStyle}
              />
            </label>

            {error && <p style={{ color: colors.status.critical, fontSize: '0.85rem', margin: '0 0 1rem' }}>{error}</p>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
              <button
                type="button"
                onClick={onClose}
                style={{ ...buttonBase, border: `1px solid ${colors.cardBorder}`, background: 'transparent', color: colors.textOnPage }}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSaving}
                style={{ ...buttonBase, border: 'none', background: colors.tileBorder, color: '#fff', opacity: isSaving ? 0.6 : 1 }}
              >
                {isSaving ? 'Saving…' : 'Change password'}
              </button>
            </div>
          </form>
        )}
      </div>
    </>,
    document.body
  );
}

export default ChangePasswordDialog;
