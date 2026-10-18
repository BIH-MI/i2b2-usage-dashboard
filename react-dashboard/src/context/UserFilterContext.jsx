import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';
import { useTheme } from '../theme';

// user selection shared by all dashboard pages, wraps them in App.jsx (not the login page) so every page reads the same state
const UserFilterContext = createContext(null);

// same number for the same text every time, so a user keeps their alias after a reload
function hashString(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

// alice -> User 7. numbered in hash order, not alphabetically, so the number does not give the name away
function buildAliases(users) {
  const shuffled = [...users].sort((a, b) => hashString(a) - hashString(b) || a.localeCompare(b));
  return new Map(shuffled.map((user, i) => [user, `User ${i + 1}`]));
}

export function UserFilterProvider({ children }) {
  const { hideUsernames } = useTheme();
  const [pickedUsers, setPickedUsers] = useState([]); // users picked by hand in the user filter, empty = everyone
  const [excludedGroups, setExcludedGroups] = useState([]); // group names currently toggled "excluded"
  const [allUsers, setAllUsers] = useState([]);
  const [groups, setGroups] = useState([]); // [{ user_id, group_name }, ...]

  useEffect(() => {
    fetchWithTimeout('/api/users')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setAllUsers(rows.map((r) => r.user_id)))
      .catch(() => setAllUsers([]));

    fetchWithTimeout('/api/user-groups')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setGroups(rows))
      .catch(() => setGroups([]));
  }, []);

  const aliases = useMemo(() => buildAliases(allUsers), [allUsers]);

  // the name to show for a user, the alias while "Usernames hidden" is on
  const displayName = useCallback(
    (userId) => (hideUsernames ? aliases.get(userId) ?? 'Unknown user' : userId),
    [hideUsernames, aliases]
  );

  const groupNames = useMemo(() => [...new Set(groups.map((g) => g.group_name))].sort(), [groups]);

  // the users the charts get: the picked users (or everyone if none picked) minus the excluded groups
  const selectedUsers = useMemo(() => {
    if (excludedGroups.length === 0) return pickedUsers;
    const excludedIds = new Set(groups.filter((g) => excludedGroups.includes(g.group_name)).map((g) => g.user_id));
    const base = pickedUsers.length > 0 ? pickedUsers : allUsers;
    return base.filter((id) => !excludedIds.has(id));
  }, [pickedUsers, excludedGroups, groups, allUsers]);

  // a change in the user filter starts from what is shown, so the excluded users stay out and the chips are cleared
  function setSelectedUsers(users) {
    setExcludedGroups([]);
    setPickedUsers(users);
  }

  // toggles one group in/out of the excluded set, re-clicking brings its members back
  function toggleExcludeGroup(groupName) {
    setExcludedGroups((current) =>
      current.includes(groupName) ? current.filter((g) => g !== groupName) : [...current, groupName]
    );
  }

  // removes only the group exclusions, the picked users stay
  function resetExcludedGroups() {
    setExcludedGroups([]);
  }

  const value = useMemo(
    () => ({
      selectedUsers, setSelectedUsers, allUsers, groupNames, excludedGroups, toggleExcludeGroup, resetExcludedGroups,
      hideUsernames, aliases, displayName
    }),
    [selectedUsers, allUsers, groupNames, excludedGroups, hideUsernames, aliases, displayName]
  );

  return <UserFilterContext.Provider value={value}>{children}</UserFilterContext.Provider>;
}

export function useUserFilter() {
  const ctx = useContext(UserFilterContext);
  if (!ctx) throw new Error('useUserFilter must be used inside <UserFilterProvider>');
  return ctx;
}
