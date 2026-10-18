import { useState, useEffect } from 'react';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';

// flat list of user_id strings, shared by UserFilter and QueriesByUser
export function useAllUsers() {
  const [allUsers, setAllUsers] = useState([]);

  useEffect(() => {
    fetchWithTimeout('/api/users')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setAllUsers(rows.map((r) => r.user_id)))
      .catch(() => setAllUsers([]));
  }, []);

  return allUsers;
}
