import { useState, useEffect } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import { buildDateRangeParams } from '../../utils/dateRangeParams';
import { fetchWithTimeout } from '../../utils/fetchWithTimeout';
import LoadingState from '../LoadingState';
import QueryLinkList from '../QueryLinkList';

// module-level constant, the same array on every render, so the useEffect below does not rerun when no userIds are passed
const EMPTY_USER_IDS = [];

const YEARS_PER_PAGE = 3; // how many years of calendar shown at once on screen
const ROW_HEIGHT = 180; // how many pixels tall one year's calendar needs (for horizontal orientation)
const TOP_MARGIN = 140; // space at the top for the title and, below it, the color scale (so neither overlaps the month labels)

function ActivityCalendar({ startDate, endDate, userIds = EMPTY_USER_IDS }) {
  const { colors } = useTheme();
  const [dailyCounts, setDailyCounts] = useState(null); // { "2025-08-14": 3, "2025-08-15": 1, ... }
  const [years, setYears] = useState([]); // every year present in the data, sorted
  const [pageIndex, setPageIndex] = useState(0);
  const [selectedDay, setSelectedDay] = useState(null);
  const [dayQueries, setDayQueries] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!startDate || !endDate) return; // date-range fetch hasn't resolved yet
    setError(false);
    const params = buildDateRangeParams(startDate, endDate, userIds);
    fetchWithTimeout(`/api/daily-count?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => { // rows = [{ day: "2025-08-14", count: 3 }, ...], already grouped by the backend
        const counts = {};
        const yearSet = new Set();

        rows.forEach(({ day, count }) => {
          counts[day] = count;
          yearSet.add(Number(day.slice(0, 4)));
        });

        setDailyCounts(counts);
        setYears([...yearSet].sort());
        setPageIndex(0); // jump back to the first page whenever the filter changes
      })
      .catch(() => setError(true));
  }, [startDate, endDate, userIds]);


  useEffect(() => {
    if (!selectedDay) return;
    setDayQueries(null);
    fetchWithTimeout(`/api/queries?start=${selectedDay}&end=${selectedDay}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((rows) => setDayQueries(rows))
      .catch(() => setDayQueries([]));
  }, [selectedDay]);

  if (error) {
    return <p style={{ color: colors.status.critical }}>Failed to load data. Try again later.</p>;
  }

    if (!dailyCounts || years.length === 0) {
    return <LoadingState height="360px" />;
  }
  // builds an array of [date, count] pairs for one specific year, for the calendar heatmap
  function getDataForYear(year) {
    const start = new Date(year, 0, 1);
    const end = new Date(year + 1, 0, 1);
    const dayMs = 24 * 60 * 60 * 1000;
    const data = [];

    for (let t = +start; t < +end; t += dayMs) {
      const day = new Date(t).toISOString().slice(0, 10);
      data.push([day, dailyCounts[day] || 0]);
    }
    return data;
  }

  // highest daily value across all days, so the color scale (visualMap) is consistent across every page
  const maxCount = Math.max(1, ...Object.values(dailyCounts));

  const totalPages = Math.ceil(years.length / YEARS_PER_PAGE);
  const visibleYears = years.slice(pageIndex * YEARS_PER_PAGE, pageIndex * YEARS_PER_PAGE + YEARS_PER_PAGE);
  const chartHeight = TOP_MARGIN + visibleYears.length * ROW_HEIGHT;

  const option = {
    title: { text: 'Activity Calendar', textStyle: { color: colors.textOnPage } }, // centered by Chart.jsx, same look as the other charts
    tooltip: {
      formatter: (p) => `${p.data[0]}: ${p.data[1]} queries`
    },
    visualMap: {
      min: 0,
      max: maxCount,
      calculable: true,
      orient: 'horizontal',
      left: 60, // left side, in line with the calendar below
      top: 40,
      inRange: {
        color: colors.sequentialBlue // light to dark, from the palette's sequential blue ramp
      },
      textStyle: { color: colors.textOnPage } // visualMap's "min"/"max" numbers
    },
    calendar: visibleYears.map((year, i) => ({
      orient: 'horizontal', // weeks left to right, days top to bottom, "GitHub-style"
      top: TOP_MARGIN + i * ROW_HEIGHT,
      left: 60,
      right: 20,
      height: ROW_HEIGHT - 30, // leave a bit of room to separate the years/labels
      range: String(year),
      dayLabel: { margin: 5, color: colors.textOnPage },
      monthLabel: { color: colors.textOnPage },
      yearLabel: { show: true, color: colors.textOnPage }
    })),
    series: visibleYears.map((year, i) => ({
      type: 'heatmap',
      coordinateSystem: 'calendar',
      calendarIndex: i,
      data: getDataForYear(year)
    }))
  };

  return (
    <div>
      <Chart
        option={option}
        style={{ height: `${chartHeight}px`, width: '100%', maxWidth: '1400px', margin: '0 auto' }} // capped so it doesn't stretch on wide screens
        onEvents={{ click: (params) => setSelectedDay(params.data[0]) }}
      />

      {/* year pages, under the calendar now that the title is drawn inside the chart */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: colors.textOnPage, fontSize: '0.85rem' }}>
          <button onClick={() => setPageIndex((p) => p - 1)} disabled={pageIndex === 0}>
            Prev
          </button>
          <span>{visibleYears.join(' - ')}</span>
          <button onClick={() => setPageIndex((p) => p + 1)} disabled={pageIndex >= totalPages - 1}>
            Next
          </button>
        </div>
      )}

      {selectedDay && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={dayQueries} title={selectedDay} />
        </div>
      )}
    </div>
  );
}

export default ActivityCalendar;
