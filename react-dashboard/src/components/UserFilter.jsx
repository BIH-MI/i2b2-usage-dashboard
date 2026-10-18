import { useState } from 'react';
import { createPortal } from 'react-dom';
import { Filter, Search, X, Check } from 'lucide-react';
import { useTheme } from '../theme';
import { useUserFilter } from '../context/UserFilterContext';

// trigger matches the sidebar nav links, clicking it slides a panel in from the right edge
function UserFilter({ isCollapsed = false }) {
  const { colors } = useTheme();
  const ACCENT_SOFT = `${colors.tileBorder}1a`;
  const ACCENT_BORDER_SOFT = `${colors.tileBorder}33`;
  const { selectedUsers, setSelectedUsers: onChange, allUsers, hideUsernames, displayName } = useUserFilter();
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  // searches the shown name. with hidden usernames the list is sorted by alias, alphabetical would give the names away
  const filteredOptions = allUsers
    .filter((u) => displayName(u).toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => (hideUsernames ? displayName(a).localeCompare(displayName(b), undefined, { numeric: true }) : 0));

  function toggleUser(user) {
    onChange(
      selectedUsers.includes(user) ? selectedUsers.filter((u) => u !== user) : [...selectedUsers, user]
    );
  }

  function selectAll() {
    onChange(allUsers);
  }

  function unselectAll() {
    onChange([]);
  }

  return (
    <>
      <style>{`
        .uf-trigger { transition: background-color 0.15s ease; }
        .uf-trigger:hover { background-color: ${colors.tileBorder}22; }
        .uf-shortcut { transition: background-color 0.15s ease, color 0.15s ease; }
        .uf-shortcut:hover { background-color: ${ACCENT_SOFT}; }
        .uf-option { transition: background-color 0.12s ease; }
        .uf-option:hover { background-color: ${ACCENT_SOFT}; }
        .uf-chip { transition: background-color 0.12s ease, border-color 0.12s ease; }
        .uf-chip:hover { border-color: ${colors.tileBorder} !important; }
        .uf-search:focus { border-color: ${colors.tileBorder} !important; }
        .uf-close:hover { background-color: ${ACCENT_SOFT}; }
      `}</style>

      <div
        className="uf-trigger"
        onClick={() => setIsOpen(true)}
        title={isCollapsed ? 'Select Users' : undefined}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: isCollapsed ? 'center' : 'flex-start',
          gap: '0.6rem',
          padding: '0.5rem 0.75rem',
          borderRadius: '6px',
          cursor: 'pointer',
          color: colors.textOnPage,
          fontSize: '0.9rem'
        }}
      >
        <Filter size={18} />
        {!isCollapsed && (
          <span style={{ flex: 1 }}>{selectedUsers.length === 0 ? 'Select Users' : `${selectedUsers.length} user${selectedUsers.length > 1 ? 's' : ''}`}</span>
        )}
      </div>

      {isOpen && createPortal(
        <>
          {/* backdrop - click anywhere outside the panel to close it */}
          <div
            onClick={() => setIsOpen(false)}
            style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.35)', zIndex: 999 }}
          />

          {/* flyout panel, floating just to the right of the sidebar (never covering it)
              and vertically centered rather than full-height */}
          <div
            style={{
              position: 'fixed',
              top: '50%',
              left: isCollapsed ? '68px' : '272px',
              transform: 'translateY(-50%)',
              height: '640px',
              maxHeight: '80vh',
              width: '480px',
              maxWidth: 'calc(90vw - 68px)',
              display: 'flex',
              overflow: 'hidden',
              backgroundColor: colors.chartSurface,
              border: `1px solid ${ACCENT_BORDER_SOFT}`,
              borderRadius: '16px',
              boxShadow: '0 16px 40px rgba(0,0,0,0.24), 0 2px 8px rgba(0,0,0,0.12)',
              zIndex: 1000
            }}
          >
            <div style={{ width: '280px', borderRight: `1px solid ${ACCENT_BORDER_SOFT}`, display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.9rem 0.9rem 0.4rem' }}>
                <span style={{ color: colors.textOnPage, fontWeight: 600, fontSize: '0.95rem' }}>Select Users</span>
                <X
                  className="uf-close"
                  size={18}
                  color={`${colors.textOnPage}99`}
                  onClick={() => setIsOpen(false)}
                  style={{ cursor: 'pointer', borderRadius: '6px', padding: '0.2rem' }}
                />
              </div>

              <div style={{ position: 'relative', padding: '0.6rem 0.9rem 0.4rem' }}>
                <Search size={15} color={`${colors.textOnPage}99`} style={{ position: 'absolute', left: '1.4rem', top: '50%', transform: 'translateY(-50%)' }} />
                <input
                  className="uf-search"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search users..."
                  autoFocus
                  style={{
                    width: '100%',
                    boxSizing: 'border-box',
                    border: `1.5px solid ${ACCENT_BORDER_SOFT}`,
                    borderRadius: '8px',
                    outline: 'none',
                    padding: '0.45rem 0.6rem 0.45rem 2rem',
                    backgroundColor: 'transparent',
                    color: colors.textOnPage,
                    fontSize: '0.85rem',
                    transition: 'border-color 0.15s ease'
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '0.4rem', padding: '0 0.9rem 0.5rem' }}>
                <span
                  className="uf-shortcut"
                  onClick={selectAll}
                  style={{ cursor: 'pointer', color: colors.tileBorder, fontSize: '0.75rem', fontWeight: 600, padding: '0.2rem 0.5rem', borderRadius: '999px' }}
                >
                  Select All
                </span>
                <span
                  className="uf-shortcut"
                  onClick={unselectAll}
                  style={{ cursor: 'pointer', color: `${colors.textOnPage}99`, fontSize: '0.75rem', fontWeight: 600, padding: '0.2rem 0.5rem', borderRadius: '999px' }}
                >
                  Unselect All
                </span>
              </div>

              <div style={{ flex: 1, overflowY: 'auto', padding: '0 0.65rem 0.65rem' }}>
                {filteredOptions.length === 0 ? (
                  <div style={{ padding: '0.6rem 0.6rem', color: `${colors.textOnPage}99`, fontSize: '0.85rem' }}>No users found</div>
                ) : (
                  filteredOptions.map((user) => {
                    const isSelected = selectedUsers.includes(user);
                    return (
                      <div
                        key={user}
                        className="uf-option"
                        onClick={() => toggleUser(user)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '0.5rem 0.6rem',
                          marginBottom: '0.15rem',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          color: colors.textOnPage,
                          fontSize: '0.85rem',
                          backgroundColor: isSelected ? ACCENT_SOFT : 'transparent'
                        }}
                      >
                        <span>{displayName(user)}</span>
                        {isSelected && <Check size={14} color={colors.tileBorder} />}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <div style={{ width: '200px', display: 'flex', flexDirection: 'column', padding: '0.9rem' }}>
              <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                Selected ({selectedUsers.length})
              </div>
              <div style={{ flex: 1, overflowY: 'auto' }}>
                {selectedUsers.length === 0 ? (
                  <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.8rem' }}>None yet</div>
                ) : (
                  selectedUsers.map((user) => (
                    <div
                      key={user}
                      className="uf-chip"
                      onClick={() => toggleUser(user)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '0.4rem',
                        padding: '0.4rem 0.55rem',
                        marginBottom: '0.3rem',
                        borderRadius: '8px',
                        border: `1px solid ${ACCENT_BORDER_SOFT}`,
                        backgroundColor: ACCENT_SOFT,
                        color: colors.textOnPage,
                        fontSize: '0.8rem',
                        cursor: 'pointer'
                      }}
                    >
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName(user)}</span>
                      <X size={13} color={`${colors.textOnPage}99`} style={{ flexShrink: 0 }} />
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </>,
        document.body
      )}
    </>
  );
}

export default UserFilter;
