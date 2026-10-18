import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { useTheme } from '../theme';
import { datePickerFormat } from '../utils/formatDate';

// each shortcut is a button that jumps the range back N months from today
const shortcuts = [
  { label: 'Last Month', months: 1 },
  { label: 'Last 3 Months', months: 3 },
  { label: 'Last 6 Months', months: 6 },
  { label: 'Last Year', months: 12 },
  { label: 'Last 2 Years', months: 24 }
];

function DateRangeFilter({ startDate, endDate, minDate, onStartChange, onEndChange, onApplyShortcut, onAllTime }) {
  const { dateFormat } = useTheme();
  // years the year dropdown should cover, based on the real data range
  const yearsSpan = minDate ? new Date().getFullYear() - minDate.getFullYear() + 1 : 15;

  return (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
      <div>
        <div>Start Date</div>
        <DatePicker
          selected={startDate}
          onChange={onStartChange}
          minDate={minDate} // earliest date in the actual db data
          maxDate={new Date()}
          showYearDropdown
          dropdownMode="select"
          yearDropdownItemNumber={yearsSpan}
          dateFormat={datePickerFormat(dateFormat)}
          calendarStartDay={dateFormat === 'eu' ? 1 : 0} // week starts on Monday in Europe, Sunday in the US
        />
      </div>
      <div>
        <div>End Date</div>
        <DatePicker
          selected={endDate}
          onChange={onEndChange}
          minDate={minDate}
          maxDate={new Date()}
          showYearDropdown
          dropdownMode="select"
          yearDropdownItemNumber={yearsSpan}
          dateFormat={datePickerFormat(dateFormat)}
          calendarStartDay={dateFormat === 'eu' ? 1 : 0} // week starts on Monday in Europe, Sunday in the US
        />
      </div>

      
      <button onClick={onAllTime}>All Time</button>

      
      {shortcuts.map((s) => (
        <button key={s.label} onClick={() => onApplyShortcut(s.months)}>
          {s.label}
        </button>
      ))}
    </div>
  );
}

export default DateRangeFilter;
