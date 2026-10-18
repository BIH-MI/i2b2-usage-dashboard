import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { fetchWithTimeout } from '../utils/fetchWithTimeout';

// date range shared by all dashboard pages, wraps them in App.jsx so the picked range stays when switching pages
const DateRangeContext = createContext(null);

export function DateRangeProvider({ children }) {
  const [startDate, setStartDate] = useState(null); //null until /api/date-range loads and the real min date is known
  const [endDate, setEndDate] = useState(null);
  const [dataRange, setDataRange] = useState(null); // { min, max }, the earliest/latest create_date
  const [error, setError] = useState(false);

  useEffect(() => {
    fetchWithTimeout('/api/date-range')
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then(({ min, max }) => {
        const minDate = new Date(min);
        const maxDate = new Date(max);
        setDataRange({ min: minDate, max: maxDate });
        setStartDate(minDate);
        setEndDate(maxDate);
      })
      .catch(() => setError(true));
  }, []);

  const value = useMemo(() => {
    function applyShortcut(monthsBack) {
      const end = new Date();
      const start = new Date();
      start.setMonth(start.getMonth() - monthsBack);
      setStartDate(start);
      setEndDate(end);
    }

    function showAllTime() {
      if (!dataRange) return;
      setStartDate(dataRange.min);
      setEndDate(dataRange.max);
    }

    return { startDate, endDate, dataRange, error, setStartDate, setEndDate, applyShortcut, showAllTime };
  }, [startDate, endDate, dataRange, error]);

  return <DateRangeContext.Provider value={value}>{children}</DateRangeContext.Provider>;
}

export function useDateRange() {
  const ctx = useContext(DateRangeContext);
  if (!ctx) throw new Error('useDateRange must be used inside <DateRangeProvider>');
  return ctx;
}
