// date and time in the format chosen in the sidebar, de-DE gives 10.10.2026, 14:30:00
export function formatDateTime(value, dateFormat) {
  return new Date(value).toLocaleString(dateFormat === 'eu' ? 'de-DE' : 'en-US');
}

// date-fns pattern for react-datepicker
export function datePickerFormat(dateFormat) {
  return dateFormat === 'eu' ? 'dd.MM.yyyy' : 'MM/dd/yyyy';
}

// uses local getters, not toISOString(), so the timezone offset never shifts the date.
// day-only, for ?start=/&end= query params
export function toDateParam(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}
