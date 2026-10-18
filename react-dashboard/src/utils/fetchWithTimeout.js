const TIMEOUT_MS = 8000;
// adds the base path, '/api/users' -> '/usage/api/users' on the server
export function apiUrl(path) {
  return import.meta.env.BASE_URL + path.replace(/^\//, '');
}


export function fetchWithTimeout(url, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(apiUrl(url), { signal: controller.signal })
    .then((res) => {
      // no session cookie or it expired, go to the login page (unless already there, that would reload forever)
      // /api/me is skipped, RequireAuth handles it and remembers the page the user wanted
      if (res.status === 401 && url !== '/api/me' && window.location.pathname !== import.meta.env.BASE_URL + 'login') window.location.assign(import.meta.env.BASE_URL + 'login');
      return res;
    })
    .finally(() => clearTimeout(timer));
}

// abort only stops the frontend, the backend may still finish after 8 seconds but nobody receives the answer
