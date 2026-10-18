import { useEffect, useState } from 'react';
import { Trash2, UserPlus } from 'lucide-react';
import ChartCard from '../components/ChartCard';
import LoadingState from '../components/LoadingState';
import RoleSelect from '../components/RoleSelect';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';
import { useCurrentUser } from '../hooks/useCurrentUser';
import { useTheme } from '../theme';
import { apiUrl } from '../utils/fetchWithTimeout';

// same rules as NewUserRequest in the backend (models.py)
const USERNAME_PATTERN = /^[A-Za-z0-9._-]+$/;
const MIN_PASSWORD_LENGTH = 8;
const MAX_PASSWORD_LENGTH = 64;
const MAX_PASSWORD_BYTES = 72; // bcrypt limit, counted in bytes, not characters

// the backend sends its reason in "detail", falls back to the given text if there is none
async function readDetail(res, fallback) {
  try {
    const body = await res.json();
    return typeof body.detail === 'string' ? body.detail : fallback;
  } catch {
    return fallback;
  }
}

// admin page: lists the dashboard users, changes their role, adds or deletes them
function SettingsPage() {
  const { colors } = useTheme();
  const currentUser = useCurrentUser(); // the own row cannot be changed or deleted
  const [users, setUsers] = useState(null); // null = still loading
  const [listError, setListError] = useState('');
  const [pendingDelete, setPendingDelete] = useState(null); // username waiting for the "Delete?" answer

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [formError, setFormError] = useState('');
  const [formSuccess, setFormSuccess] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  function loadUsers() {
    fetchWithTimeout('/api/admin/users')
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 403 ? 'Only admins can manage users.' : 'Could not load the users.');
        return res.json();
      })
      .then((rows) => {
        setUsers(rows);
        setListError('');
      })
      .catch((err) => {
        setUsers([]);
        setListError(err.name === 'AbortError' || !err.message ? 'Could not load the users.' : err.message);
      });
  }

  useEffect(() => {
    loadUsers();
  }, []);

  async function handleAdd(e) {
    e.preventDefault();
    setFormError('');
    setFormSuccess('');

    const name = username.trim();
    if (!name) {
      setFormError('Enter a username.');
      return;
    }
    if (!USERNAME_PATTERN.test(name)) {
      setFormError('The username may only contain letters, digits, dot, underscore and hyphen.');
      return;
    }
    if (password.length < MIN_PASSWORD_LENGTH) {
      setFormError(`The password needs at least ${MIN_PASSWORD_LENGTH} characters.`);
      return;
    }
    if (password.length > MAX_PASSWORD_LENGTH || new TextEncoder().encode(password).length > MAX_PASSWORD_BYTES) {
      setFormError('The password is too long.');
      return;
    }

    setIsSaving(true);
    try {
      const res = await fetch(apiUrl('/api/admin/users'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: name, password, role })
      });

      if (res.ok) {
        setFormSuccess(`Added ${name}.`);
        setUsername('');
        setPassword('');
        setRole('user');
        loadUsers();
      } else if (res.status === 409) {
        setFormError('This username already exists.');
      } else if (res.status === 403) {
        setFormError('Only admins can add users.');
      } else {
        setFormError(await readDetail(res, 'The user could not be added.'));
      }
    } catch {
      setFormError('The user could not be added.');
    } finally {
      setIsSaving(false);
    }
  }

  async function handleRoleChange(name, newRole) {
    setListError('');
    try {
      const res = await fetch(apiUrl(`/api/admin/users/${encodeURIComponent(name)}`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole })
      });
      if (!res.ok) setListError(await readDetail(res, `The role of ${name} could not be changed.`));
    } catch {
      setListError(`The role of ${name} could not be changed.`);
    }
    loadUsers(); // also after a failure, so the dropdown shows what is really stored
  }

  async function handleDelete(name) {
    setPendingDelete(null);
    setListError('');
    try {
      const res = await fetch(apiUrl(`/api/admin/users/${encodeURIComponent(name)}`), { method: 'DELETE' });
      if (res.ok) {
        loadUsers();
      } else {
        setListError(await readDetail(res, `${name} could not be deleted.`));
      }
    } catch {
      setListError(`${name} could not be deleted.`);
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
    marginBottom: '0.35rem',
    color: colors.textOnPage,
    fontSize: '0.8rem',
    fontWeight: 600
  };

  const cardTitleStyle = { color: 'var(--text-h)', fontSize: '1rem', fontWeight: 600, margin: '0 0 1rem' };

  return (
    <div style={{ padding: '2rem' }}>
      <h2 style={{ color: colors.textOnPage, marginBottom: '1rem' }}>Settings</h2>

      {/* hover can't be set in inline styles */}
      <style>{`
        .settings-user-row { transition: background 0.15s; }
        .settings-user-row:hover { background: ${colors.tileBorder}12; }
      `}</style>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-start' }}>
        <div style={{ flex: '2 1 420px', minWidth: 0 }}>
          <ChartCard info="Everyone who can sign in to this dashboard. Admins can also open this page and manage the users.">
            <h3 style={cardTitleStyle}>Dashboard users{users && users.length > 0 ? ` (${users.length})` : ''}</h3>

            {listError && (
              <p style={{ color: colors.status.critical, fontSize: '0.85rem', margin: '0 0 0.75rem' }}>{listError}</p>
            )}

            {!users ? (
              <LoadingState height="80px" />
            ) : users.length === 0 ? (
              !listError && <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No users yet.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                {users.map((u) => {
                  const isAdmin = u.role === 'admin';
                  const isPending = pendingDelete === u.username;
                  const isSelf = currentUser?.username === u.username;
                  const roleStyle = {
                    flexShrink: 0,
                    padding: '0.15rem 0.6rem',
                    borderRadius: '999px',
                    background: isAdmin ? `${colors.tileBorder}1a` : 'transparent',
                    border: `1px solid ${isAdmin ? 'transparent' : colors.cardBorder}`,
                    color: isAdmin ? colors.tileBorder : colors.textOnPage,
                    fontSize: '0.72rem',
                    fontWeight: 600
                  };
                  return (
                    <div
                      key={u.username}
                      className="settings-user-row"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.6rem',
                        padding: '0.5rem 0.7rem',
                        borderRadius: '10px',
                        border: `1px solid ${isPending ? `${colors.status.critical}88` : colors.cardBorder}`,
                        color: colors.textOnPage,
                        fontSize: '0.88rem'
                      }}
                    >
                      <span
                        title={u.username}
                        style={{ flex: 1, minWidth: 0, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      >
                        {u.username}
                        {isSelf && <span style={{ color: `${colors.textOnPage}99`, fontWeight: 400 }}> (you)</span>}
                      </span>

                      {/* the own role is fixed, otherwise the last admin could lock everyone out */}
                      {isSelf ? (
                        <span style={roleStyle}>{isAdmin ? 'Admin' : 'User'}</span>
                      ) : (
                        <RoleSelect
                          variant="pill"
                          value={u.role}
                          onChange={(newRole) => handleRoleChange(u.username, newRole)}
                          title={`Change the role of ${u.username}`}
                        />
                      )}

                      {isSelf ? null : isPending ? (
                        <span style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <span style={{ fontSize: '0.78rem' }}>Delete?</span>
                          <button
                            type="button"
                            onClick={() => handleDelete(u.username)}
                            style={{ background: colors.status.critical, border: 'none', color: '#fff', fontWeight: 600 }}
                          >
                            Delete
                          </button>
                          <button type="button" onClick={() => setPendingDelete(null)}>
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <span
                          title={`Delete ${u.username}`}
                          onClick={() => setPendingDelete(u.username)}
                          style={{ flexShrink: 0, display: 'inline-flex', cursor: 'pointer' }}
                        >
                          <Trash2 size={16} color={`${colors.textOnPage}99`} />
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </ChartCard>
        </div>

        <div style={{ flex: '1 1 300px', minWidth: 0 }}>
          <ChartCard info="The password is hashed before it is stored, nobody can read it afterwards. Give it to the new user in person.">
            <h3 style={cardTitleStyle}>Add user</h3>

            <form onSubmit={handleAdd}>
              <label style={{ display: 'block', marginBottom: '0.9rem' }}>
                <span style={labelSpanStyle}>Username</span>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  maxLength={64}
                  autoComplete="off"
                  style={inputStyle}
                />
              </label>

              <label style={{ display: 'block', marginBottom: '0.9rem' }}>
                <span style={labelSpanStyle}>Password</span>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
                  style={inputStyle}
                />
                <span style={{ display: 'block', marginTop: '0.3rem', color: `${colors.textOnPage}99`, fontSize: '0.72rem' }}>
                  At least {MIN_PASSWORD_LENGTH} characters
                </span>
              </label>

              {/* a div, not a label: inside a label a click on an option would reopen the list */}
              <div style={{ marginBottom: '1.1rem' }}>
                <span style={labelSpanStyle}>Role</span>
                <RoleSelect value={role} onChange={setRole} title="Role of the new user" />
              </div>

              {formError && (
                <p style={{ color: colors.status.critical, fontSize: '0.85rem', margin: '0 0 0.9rem' }}>{formError}</p>
              )}
              {formSuccess && (
                <p style={{ color: colors.status.good, fontSize: '0.85rem', margin: '0 0 0.9rem' }}>{formSuccess}</p>
              )}

              <button
                type="submit"
                disabled={isSaving}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.4rem',
                  padding: '0.6rem',
                  borderRadius: '10px',
                  border: 'none',
                  background: colors.tileBorder,
                  color: '#fff',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  boxShadow: `0 4px 12px ${colors.tileBorder}44`
                }}
              >
                <UserPlus size={16} /> {isSaving ? 'Adding...' : 'Add user'}
              </button>
            </form>
          </ChartCard>
        </div>
      </div>
    </div>
  );
}

export default SettingsPage;
