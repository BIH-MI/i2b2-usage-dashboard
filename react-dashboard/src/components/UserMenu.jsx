import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { ChevronDown, Settings, LogOut, KeyRound } from 'lucide-react';
import { useTheme } from '../theme';
import { useCurrentUser } from '../hooks/useCurrentUser';
import { apiUrl } from '../utils/fetchWithTimeout';
import ChangePasswordDialog from './ChangePasswordDialog';

// asks before logging out, same backdrop and card look as the user filter panel
function LogoutConfirm({ onCancel, onConfirm }) {
  const { colors } = useTheme();

  // Escape closes it, like clicking the backdrop
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') onCancel();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onCancel]);

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
        onClick={onCancel}
        style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.35)', zIndex: 999 }}
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="logout-confirm-title"
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '340px',
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
        <h3 id="logout-confirm-title" style={{ color: 'var(--text-h)', margin: '0 0 0.4rem', fontSize: '1.05rem' }}>
          Log out?
        </h3>
        <p style={{ color: colors.textOnPage, fontSize: '0.88rem', margin: '0 0 1.25rem' }}>
          You will need to sign in again to see the dashboard.
        </p>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
          <button
            type="button"
            onClick={onCancel}
            autoFocus
            style={{ ...buttonBase, border: `1px solid ${colors.cardBorder}`, background: 'transparent', color: colors.textOnPage }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            style={{ ...buttonBase, border: 'none', background: colors.tileBorder, color: '#fff' }}
          >
            Log out
          </button>
        </div>
      </div>
    </>,
    document.body
  );
}

// account button in the top right corner: who is logged in, change password, settings (admins), log out
function UserMenu() {
  const { colors } = useTheme();
  const currentUser = useCurrentUser();
  const [isOpen, setIsOpen] = useState(false);
  const [isLogoutOpen, setIsLogoutOpen] = useState(false);
  const [isPasswordOpen, setIsPasswordOpen] = useState(false);

  const username = currentUser?.username ?? '';
  const isAdmin = currentUser?.role === 'admin'; // only hides the entry, the backend does the real check

  // Escape closes the dropdown
  useEffect(() => {
    if (!isOpen) return;
    function onKeyDown(e) {
      if (e.key === 'Escape') setIsOpen(false);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen]);

  // backend deletes the session cookie, then a full page load to /login clears what is left in memory
  async function handleLogout() {
    try {
      await fetch(apiUrl('/api/logout'), { method: 'POST' });
    } finally {
      window.location.assign(import.meta.env.BASE_URL + 'login');
    }
  }

  const itemStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    padding: '0.5rem 0.6rem',
    borderRadius: '8px',
    color: colors.textOnPage,
    fontSize: '0.88rem',
    textDecoration: 'none',
    cursor: 'pointer'
  };

  const divider = <div style={{ height: '1px', background: colors.cardBorder, margin: '0.35rem 0' }} />;

  return (
    <div style={{ position: 'absolute', top: '1.5rem', right: '2rem', zIndex: 50 }}>
      {/* hover can't be set in inline styles */}
      <style>{`
        .user-menu-item { transition: background 0.15s; }
        .user-menu-item:hover { background: ${colors.tileBorder}1a; }
      `}</style>

      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        title="Account"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.3rem 0.6rem 0.3rem 0.3rem',
          borderRadius: '999px',
          border: `1px solid ${colors.cardBorder}`,
          background: colors.chartSurface,
          color: colors.textOnPage,
          fontSize: '0.85rem',
          boxShadow: colors.cardShadow
        }}
      >
        <span
          style={{
            width: '26px',
            height: '26px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: colors.tileBorder,
            color: '#fff',
            fontSize: '0.78rem',
            fontWeight: 700
          }}
        >
          {username ? username[0].toUpperCase() : '?'}
        </span>
        <span style={{ maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 600 }}>
          {username || 'Account'}
        </span>
        <ChevronDown size={15} />
      </button>

      {isOpen && (
        <>
          {/* invisible layer, a click anywhere outside closes the dropdown */}
          <div onClick={() => setIsOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 50 }} />

          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 0.4rem)',
              right: 0,
              width: '220px',
              padding: '0.4rem',
              backgroundColor: colors.chartSurface,
              border: `1px solid ${colors.cardBorder}`,
              borderRadius: '12px',
              boxShadow: '0 16px 40px rgba(0,0,0,0.24), 0 2px 8px rgba(0,0,0,0.12)',
              zIndex: 51
            }}
          >
            <div style={{ padding: '0.45rem 0.6rem 0.5rem' }}>
              <div
                title={username}
                style={{ color: 'var(--text-h)', fontSize: '0.9rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              >
                {username || 'Account'}
              </div>
              {currentUser?.role && (
                <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.75rem' }}>{currentUser.role}</div>
              )}
            </div>

            {divider}

            <div
              className="user-menu-item"
              onClick={() => {
                setIsOpen(false);
                setIsPasswordOpen(true);
              }}
              style={itemStyle}
            >
              <KeyRound size={16} /> Change password
            </div>

            {isAdmin && (
              <Link to="/settings" className="user-menu-item" onClick={() => setIsOpen(false)} style={itemStyle}>
                <Settings size={16} /> Settings
              </Link>
            )}

            {divider}

            <div
              className="user-menu-item"
              onClick={() => {
                setIsOpen(false);
                setIsLogoutOpen(true);
              }}
              style={itemStyle}
            >
              <LogOut size={16} /> Log out
            </div>
          </div>
        </>
      )}

      {isLogoutOpen && <LogoutConfirm onCancel={() => setIsLogoutOpen(false)} onConfirm={handleLogout} />}
      {isPasswordOpen && <ChangePasswordDialog onClose={() => setIsPasswordOpen(false)} />}
    </div>
  );
}

export default UserMenu;
