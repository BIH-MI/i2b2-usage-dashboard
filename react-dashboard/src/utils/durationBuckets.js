export const FAST_CUTOFF_SECONDS = 30;

export const DURATION_BUCKETS = [ // label plus max duration in seconds
  { label: '30s-1m', max: 60 },
  { label: '1-2m', max: 120 },
  { label: '2-10m', max: 600 },
  { label: '10-30m', max: 1800 },
  { label: '30m-1h', max: 3600 },
  { label: '1-2h', max: 7200 },
  { label: '2-5h', max: 18000 },
  { label: '5h+', max: Infinity }
];

// which bucket a duration falls into, null if under the fast cutoff
export function bucketFor(duration) {
  if (duration == null || duration < FAST_CUTOFF_SECONDS) return null;
  const bucket = DURATION_BUCKETS.find((b) => duration < b.max);
  return bucket ? bucket.label : DURATION_BUCKETS[DURATION_BUCKETS.length - 1].label; // fallback to the last bucket
}
