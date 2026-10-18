// helpers for the panels of one query (the `panels` field from /api/queries)

// i2b2 value operators as symbols
const VALUE_OPERATORS = { LT: '<', LE: '≤', GT: '>', GE: '≥', EQ: '=', NE: '≠', BETWEEN: 'between' };

// "2026-09-01T00:00:00" becomes "2026-09-01", a missing date shows as "…"
export function formatDay(value) {
  return value ? value.slice(0, 10) : '…';
}

// "LT", "6", "mg/dl" becomes "< 6 mg/dl". Missing parts are left out, unknown operators are shown as they are
export function formatValue(item) {
  const operator = VALUE_OPERATORS[item.value_operator] || item.value_operator;
  return [operator, item.value_constraint, item.value_unit].filter(Boolean).join(' ');
}

// true when the start date is after the end date, i.e. the dates were entered the wrong way round
export function isReversedRange(item) {
  return item.date_from && item.date_to && item.date_from > item.date_to;
}

// panels as one string, so two queries can be compared with ===.
// The name is not part of it because i2b2 adds the time to it
export function definitionKey(query) {
  return JSON.stringify(query.panels);
} 

// every constraint of the query as one row: dates and values per concept, occurrence per panel
export function constraintRows(query) {
  const rows = [];
  query.panels.forEach((panel, p) => {
    panel.items.forEach((item) => {
      // date range, one open end is enough
      if (item.date_from || item.date_to) {
        rows.push({
          type: 'Date',
          panel: p + 1,
          concepts: [item.key],
          detail: `${formatDay(item.date_from)} → ${formatDay(item.date_to)}`,
          warning: isReversedRange(item) ? 'Start is after end' : null
        });
      }
      // value limit, e.g. "< 6 mg/dl"
      if (item.value_operator) {
        rows.push({ type: 'Value', panel: p + 1, concepts: [item.key], detail: formatValue(item) });
      }
    });
    // occurrence belongs to the whole panel, so it lists all of the panel's concepts
    if (panel.occurrences > 1) {
      rows.push({ type: 'Occurrence', panel: p + 1, concepts: panel.items.map((item) => item.key), detail: `at least ${panel.occurrences} times` });
    }
  });
  return rows;
}
