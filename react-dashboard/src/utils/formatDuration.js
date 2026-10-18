// formats seconds as a short string, using the coarsest readable unit
export function formatDuration(seconds) {
  if (seconds == null) return 'still running';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
