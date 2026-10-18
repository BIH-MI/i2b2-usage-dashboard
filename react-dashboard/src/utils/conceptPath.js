// i2b2 concept paths look like \\i2b2\Diagnoses\Some Long Name\
// the last segment is the readable part, the rest is ontology plumbing
export function splitConceptPath(path) {
  const segments = path.split('\\').filter(Boolean);
  const label = segments.pop() || path;
  return { prefix: segments.join(' › '), label };
}
