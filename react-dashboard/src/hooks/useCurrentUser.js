import { useState, useEffect } from 'react';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';

// who is logged in: { username, role }, undefined while loading, null if not logged in, false if the server is not reachable
export function useCurrentUser() {
  const [currentUser, setCurrentUser] = useState(undefined);

  useEffect(() => {
    fetchWithTimeout('/api/me')
      .then((res) => {
        if (res.status === 401) return null;
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((user) => setCurrentUser(user))
      .catch(() => setCurrentUser(false));
  }, []);

  return currentUser;
}
