import * as echarts from 'echarts';
import Chart from '../Chart';
import { useTheme } from '../../theme';

const ZOOM_SIZE = 6; // how many months stay visible around a clicked bar


function DateConstraintCoverage({ queries }) {
  const { colors } = useTheme();

  const counts = {}; // "2024-01" -> number of queries whose date_constraint range covers that month
  queries.forEach((q) => {
    q.date_constraint_values.forEach(({ date_from, date_to }) => {
      if (!date_from || !date_to) return;
      const start = new Date(date_from);
      const end = new Date(date_to);
      if (isNaN(start) || isNaN(end) || start > end) return; 

      let year = start.getFullYear();
      let month = start.getMonth(); // 0-11
      const endYear = end.getFullYear();
      const endMonth = end.getMonth();

      while (year < endYear || (year === endYear && month <= endMonth)) {
        const key = `${year}-${String(month + 1).padStart(2, '0')}`; // "2024-03"
        counts[key] = (counts[key] || 0) + 1;

        month += 1;
        if (month > 11) {
          month = 0;
          year += 1;
        }
      }
    });
  });

  const seenMonths = Object.keys(counts).sort();

  if (seenMonths.length === 0) {
    return <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No date-range constraints found in this range.</p>;
  }

  
  const allMonths = [];
  let [year, month] = seenMonths[0].split('-').map(Number);
  month -= 1; // to 0-11
  const [lastYear, lastMonth1] = seenMonths[seenMonths.length - 1].split('-').map(Number);
  const lastMonth = lastMonth1 - 1; // to 0-11
  while (year < lastYear || (year === lastYear && month <= lastMonth)) {
    allMonths.push(`${year}-${String(month + 1).padStart(2, '0')}`);
    month += 1;
    if (month > 11) {
      month = 0;
      year += 1;
    }
  }

  const values = allMonths.map((m) => counts[m] || 0);

  const option = {
    title: { text: 'Most Requested Date Ranges', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: allMonths,
      axisLabel: { color: colors.textOnPage },
      axisTick: { show: false },
      axisLine: { show: false }
    },
    yAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: colors.textOnPage },
      name: 'Queries covering this month',
      nameTextStyle: { color: colors.textOnPage }
    },
    dataZoom: [{ type: 'inside' }],
    series: [
      {
        type: 'bar',
        showBackground: true,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: colors.sequentialBlue[1] },
            { offset: 0.5, color: colors.sequentialBlue[2] },
            { offset: 1, color: colors.sequentialBlue[2] }
          ])
        },
        emphasis: {
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: colors.sequentialBlue[3] },
              { offset: 0.7, color: colors.sequentialBlue[3] },
              { offset: 1, color: colors.sequentialBlue[1] }
            ])
          }
        },
        data: values
      }
    ]
  };

  return (
    <Chart
      option={option}
      style={{ height: '360px', width: '100%' }}
      onEvents={{
        click: (params, chart) => {
          chart.dispatchAction({
            type: 'dataZoom',
            startValue: allMonths[Math.max(params.dataIndex - ZOOM_SIZE / 2, 0)],
            endValue: allMonths[Math.min(params.dataIndex + ZOOM_SIZE / 2, allMonths.length - 1)]
          });
        }
      }}
    />
  );
}

export default DateConstraintCoverage;
