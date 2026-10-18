import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { ChevronDown, ChevronRight, ChevronLeft, Menu, Sun, Moon, Calendar, Eye, EyeOff } from 'lucide-react';
import { useTheme } from '../theme';
import { useUserFilter } from '../context/UserFilterContext';
import { navItems } from './navConfig';
import UserFilter from '../components/UserFilter';
import QuerySearch from '../components/QuerySearch';
import InfoTooltip from '../components/InfoTooltip';

function getLinkBaseStyle(colors) {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    padding: '0.5rem 0.75rem',
    borderRadius: '6px',
    textDecoration: 'none',
    color: colors.textOnPage,
    fontSize: '0.9rem'
  };
}

// one nav item, with or without children. When isCollapsed, only the icon shows
function NavGroup({ item, isCollapsed }) {
  const { colors } = useTheme();
  const linkBaseStyle = getLinkBaseStyle(colors);
  const hasChildren = Array.isArray(item.items) && item.items.length > 0;
  const [isExpanded, setIsExpanded] = useState(false);
  const Icon = item.icon;

  if (!hasChildren) {
    return (
      <NavLink
        to={item.href}
        title={isCollapsed ? item.label : undefined}
        style={({ isActive }) => ({
          ...linkBaseStyle,
          justifyContent: isCollapsed ? 'center' : 'flex-start',
          backgroundColor: isActive ? colors.tileBorder + '22' : 'transparent',
          fontWeight: isActive ? 600 : 400
        })}
      >
        <Icon size={18} />
        {!isCollapsed && <span style={{ flex: 1 }}>{item.label}</span>}
        {!isCollapsed && item.badge && <Badge count={item.badge} />}
      </NavLink>
    );
  }

  return (
    <div>
      <div
        onClick={() => setIsExpanded((v) => !v)}
        title={isCollapsed ? item.label : undefined}
        style={{ ...linkBaseStyle, justifyContent: isCollapsed ? 'center' : 'flex-start', cursor: 'pointer' }}
      >
        <Icon size={18} />
        {!isCollapsed && <span style={{ flex: 1 }}>{item.label}</span>}
        {!isCollapsed && (isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />)}
      </div>
      {!isCollapsed && isExpanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', marginLeft: '1.5rem' }}>
          {item.items.map((child) => {
            const ChildIcon = child.icon;
            return (
              <NavLink
                key={child.href}
                to={child.href}
                style={({ isActive }) => ({
                  ...linkBaseStyle,
                  fontSize: '0.85rem',
                  backgroundColor: isActive ? colors.tileBorder + '22' : 'transparent',
                  fontWeight: isActive ? 600 : 400
                })}
              >
                <ChildIcon size={16} />
                <span style={{ flex: 1 }}>{child.label}</span>
                {child.badge && <Badge count={child.badge} />}
              </NavLink>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Badge({ count }) {
  const { colors } = useTheme();
  return (
    <span
      style={{
        fontSize: '0.7rem',
        fontWeight: 600,
        color: '#fff',
        backgroundColor: colors.tileBorder,
        borderRadius: '999px',
        padding: '0.05rem 0.4rem'
      }}
    >
      {count}
    </span>
  );
}

// app-wide user filter, shown below the nav links so it applies on every page
function UserFilterBlock({ colors, isCollapsed }) {
  const { groupNames, excludedGroups, toggleExcludeGroup, resetExcludedGroups } = useUserFilter();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '0.75rem 0', borderTop: `1px solid ${colors.tileBorder}22`, borderBottom: `1px solid ${colors.tileBorder}22`, margin: '0.5rem 0' }}>
      <UserFilter isCollapsed={isCollapsed} />

      {!isCollapsed && groupNames.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ fontSize: '0.72rem', fontWeight: 600, color: colors.textOnPage }}>Exclude groups</span>
            <InfoTooltip
              corner={false}
              text="Groups are set per user in the i2b2 webclient admin panel, under the 'group' parameter. Click a group to exclude everyone in it from all charts. If users are selected, the group is removed from that selection."
            />
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
          {groupNames.map((groupName) => {
            const isActive = excludedGroups.includes(groupName);
            return (
              <span
                key={groupName}
                onClick={() => toggleExcludeGroup(groupName)}
                title={isActive ? `Click to include "${groupName}" again` : `Exclude everyone in "${groupName}"`}
                style={{
                  cursor: 'pointer',
                  color: isActive ? '#fff' : colors.textOnPage,
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  padding: '0.2rem 0.55rem',
                  borderRadius: '999px',
                  border: `1px solid ${colors.tileBorder}55`,
                  backgroundColor: isActive ? colors.tileBorder : 'transparent'
                }}
              >
                Exclude {groupName}
              </span>
            );
          })}
          {excludedGroups.length > 0 && (
            <span
              onClick={resetExcludedGroups}
              title="Include all groups again, the user selection stays"
              style={{
                cursor: 'pointer',
                color: colors.textSecondary,
                fontSize: '0.72rem',
                fontWeight: 600,
                padding: '0.2rem 0.55rem',
                borderRadius: '999px'
              }}
            >
              Reset
            </span>
          )}
          </div>
        </div>
      )}
    </div>
  );
}

// navigation, the filters that change what the charts show, and the display switches (date format, usernames, theme). Settings and log out are in UserMenu
function Sidebar() {
  const { colors, mode, toggleMode, dateFormat, toggleDateFormat, hideUsernames, toggleHideUsernames } = useTheme();
  const linkBaseStyle = getLinkBaseStyle(colors);
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div
      style={{
        width: isCollapsed ? '56px' : '260px',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between', // display switches stay at the bottom
        backgroundColor: colors.sidebarBg, // opaque so nothing behind it can show through
        borderRight: `1px solid ${colors.sidebarBorder}`,
        padding: '1rem 0.75rem',
        height: 'calc(100vh - 2rem)', // minus the top and bottom padding, otherwise the last row ends below the screen
        position: 'sticky',
        top: 0,
        zIndex: 30,
        transition: 'width 0.15s ease'
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <div
          onClick={() => setIsCollapsed((v) => !v)}
          title={isCollapsed ? 'Expand menu' : 'Collapse menu'}
          style={{ ...linkBaseStyle, justifyContent: isCollapsed ? 'center' : 'flex-start', cursor: 'pointer', marginBottom: '0.5rem' }}
        >
          {isCollapsed ? <Menu size={18} /> : <ChevronLeft size={18} />}
          {!isCollapsed && <span style={{ flex: 1, color: colors.textSecondary, fontSize: '0.8rem' }}>Collapse</span>}
        </div>

        {navItems.map((item) => (
          <NavGroup key={item.href} item={item} isCollapsed={isCollapsed} />
        ))}

        <UserFilterBlock colors={colors} isCollapsed={isCollapsed} />

        <QuerySearch isCollapsed={isCollapsed} />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginBottom: '0.75rem' }}>
        {/* shows the current format, a click switches to the other one */}
        <div
          onClick={toggleDateFormat}
          title={dateFormat === 'eu' ? 'Switch to US date format (MM/DD/YYYY)' : 'Switch to European date format (DD.MM.YYYY)'}
          style={{ ...linkBaseStyle, justifyContent: isCollapsed ? 'center' : 'flex-start', cursor: 'pointer' }}
        >
          <Calendar size={18} />
          {!isCollapsed && <span style={{ flex: 1 }}>{dateFormat === 'eu' ? 'DD.MM.YYYY' : 'MM/DD/YYYY'}</span>}
        </div>

        <div
          onClick={toggleHideUsernames}
          title={hideUsernames ? 'Usernames are shown as User 1, User 2... on every page and in downloaded PNGs. Click to show the real names' : 'Click to show User 1, User 2... instead of usernames, e.g. for screen sharing'}
          style={{ ...linkBaseStyle, justifyContent: isCollapsed ? 'center' : 'flex-start', cursor: 'pointer' }}
        >
          {hideUsernames ? <EyeOff size={18} /> : <Eye size={18} />}
          {!isCollapsed && <span style={{ flex: 1 }}>{hideUsernames ? 'Usernames hidden' : 'Usernames shown'}</span>}
        </div>

        <div
          onClick={toggleMode}
          title={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          style={{ ...linkBaseStyle, justifyContent: isCollapsed ? 'center' : 'flex-start', cursor: 'pointer' }}
        >
          {mode === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          {!isCollapsed && <span style={{ flex: 1 }}>{mode === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
