// colors read via useTheme(), so no hex is hardcoded elsewhere

import { createContext, useContext, useEffect, useMemo, useState } from 'react';

// StatTile stays a fixed light card regardless of theme
const FIXED = {
  textPrimary: '#0b0b0b',   // dark text on light backgrounds, e.g. inside StatTile
  textSecondary: '#52514e', // gray text on light backgrounds, for labels
  tileBackground: '#fcfcfb', // StatTile background

  sequentialBlue: ['#e3eefc', '#6da7ec', '#256abf', '#104281'], // heatmap/calendar gradient, light to dark

  status: { // fixed severity palette, distinct from the categorical hues
    good: '#0ca30c',      // FINISHED, COMPLETED
    warning: '#fab219',   // CANCELLED, INCOMPLETE
    serious: '#ec835a',   // TIMEDOUT
    critical: '#d03b3b',  // ERROR
    neutral: '#898781'    // QUEUED, PROCESSING, MEDIUM_QUEUE, LARGE_QUEUE
  }
};

// page text and categorical/accent hues flip between light and dark
const THEMED = {
  light: {
    textOnPage: '#374151',
    tileBorder: '#2a78d6',
    chartSurface: '#ffffff', // white cards on the gray page (--bg in index.css)
    cardBorder: '#e2e6ee', // neutral border, the blue one looked heavy on white
    cardShadow: '0 1px 2px rgba(16, 24, 40, 0.04), 0 4px 12px rgba(16, 24, 40, 0.06)',
    sidebarBg: '#eaf1fb', // soft blue, so the page isn't all white
    sidebarBorder: '#d6e2f3',
    gridLine: '#e6eaf0',
    categorical: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'] // CVD-safe
  },
  dark: {
    textOnPage: '#a3aab8',
    tileBorder: '#3987e5',
    chartSurface: '#171a21', // cool dark card, a step lighter than the page (--bg in index.css)
    cardBorder: '#ffffff12',
    cardShadow: '0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px rgba(0, 0, 0, 0.25)',
    sidebarBg: '#121a2b',
    sidebarBorder: '#1f2940', // quiet divider instead of the bright blue line
    gridLine: '#ffffff14', // chart grid lines, dim so they don't compete with the data
    categorical: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
  }
};

function buildColors(mode) {
  return { ...FIXED, ...THEMED[mode] };
}

const ThemeContext = createContext(null);

function getInitialMode() {
  const stored = typeof window !== 'undefined' ? window.localStorage.getItem('theme-mode') : null;
  if (stored === 'light' || stored === 'dark') return stored;
  // defaults to dark
  return 'dark';
}

// shows User 1, User 2... instead of the usernames everywhere, also in downloaded PNGs. off by default
function getInitialHideUsernames() {
  const stored = typeof window !== 'undefined' ? window.localStorage.getItem('hide-usernames') : null;
  return stored === 'true';
}

// 'eu' = DD.MM.YYYY (Germany), 'us' = MM/DD/YYYY
function getInitialDateFormat() {
  const stored = typeof window !== 'undefined' ? window.localStorage.getItem('date-format') : null;
  if (stored === 'eu' || stored === 'us') return stored;
  return navigator.language === 'en-US' ? 'us' : 'eu'; // default from the browser language
}

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState(getInitialMode);
  const [dateFormat, setDateFormat] = useState(getInitialDateFormat);
  const [hideUsernames, setHideUsernames] = useState(getInitialHideUsernames);

  useEffect(() => {
    window.localStorage.setItem('theme-mode', mode);
    document.documentElement.dataset.theme = mode; // lets index.css's [data-theme] rules override prefers-color-scheme
  }, [mode]);

  useEffect(() => {
    window.localStorage.setItem('date-format', dateFormat);
  }, [dateFormat]);

  useEffect(() => {
    window.localStorage.setItem('hide-usernames', String(hideUsernames));
  }, [hideUsernames]);

  const value = useMemo(
    () => ({
      mode,
      colors: buildColors(mode),
      toggleMode: () => setMode((m) => (m === 'dark' ? 'light' : 'dark')),
      dateFormat,
      toggleDateFormat: () => setDateFormat((f) => (f === 'eu' ? 'us' : 'eu')),
      hideUsernames,
      toggleHideUsernames: () => setHideUsernames((h) => !h)
    }),
    [mode, dateFormat, hideUsernames]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>');
  return ctx;
}
