import Chart from '../Chart';
import { useTheme } from '../../theme';
import { DURATION_BUCKETS, FAST_CUTOFF_SECONDS, bucketFor } from '../../utils/durationBuckets';
import QueryLinkList from '../QueryLinkList';
import { useState } from 'react';

function DurationHistogram({ queries }) {
  const { colors } = useTheme();
  const [selectedBucket, setSelectedBucket] = useState(null);

  const counts = DURATION_BUCKETS.map(() => 0);
  const bucket_queries = DURATION_BUCKETS.map(() => []);
  let running = 0; // no end date yet
  let fast = 0; // under the cutoff, not part of the "slow" classes shown here
  
  queries.forEach((q) => {
    if (q.duration == null) {
      running += 1;
      return;
    }
    const label = bucketFor(q.duration);
    if (label === null) {
      fast += 1;
      return;
    }
    counts[DURATION_BUCKETS.findIndex((b) => b.label === label)] += 1;
    bucket_queries[DURATION_BUCKETS.findIndex((b) => b.label === label)].push(q);

  
  
  });

  const option = {
    title: { text: 'Slow Query Duration Classes (30s+)', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'axis' },
    grid: { top: 90, bottom: 40, containLabel: true }, // pushes the bars further down from the title and tightens the empty space below
    xAxis: {
      type: 'category',
      data: DURATION_BUCKETS.map((b) => b.label),
      axisLabel: { color: colors.textOnPage }
    },
    yAxis: {
      type: 'value',
      name: 'Queries',
      axisLabel: { color: colors.textOnPage },
      nameTextStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 60,
        data: counts,
        itemStyle: { color: colors.categorical[0] }
      }
    ]
  };

  return (
    <div>
      <Chart
        option={option}
        style={{ height: '320px', width: '100%' }}
        onEvents={{ click: (params) => setSelectedBucket(params.name) }}
      />
      <p style={{ color: `${colors.textOnPage}99`, fontSize: '0.8rem', marginTop: '0.25rem' }}>
        {fast} quer{fast === 1 ? 'y' : 'ies'} finished under {FAST_CUTOFF_SECONDS}s (not shown){running > 0 && `, ${running} without end date`}.
      </p>

      {selectedBucket && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList
            queries={bucket_queries[DURATION_BUCKETS.findIndex((b) => b.label === selectedBucket)]}
            title={selectedBucket}
          />
        </div>
      )}
    </div>
  );
}

export default DurationHistogram;
