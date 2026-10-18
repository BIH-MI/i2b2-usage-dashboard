import { useState } from 'react';
import Chart from '../Chart';
import { useTheme } from '../../theme';
import QueryLinkList from '../QueryLinkList';

// every concept touch is either an inclusion or exclusion, mutually exclusive,
// so unlike the result-type split this one sums to 100%
function InclusionExclusionSplit({ queries }) {
  const { colors } = useTheme();
  const [selectedSplit, setSelectedSplit] = useState(null); // 'Included' or 'Excluded' slice, if clicked

  let included = 0;
  let excluded = 0;
  // a query can land in both lists at once, it's not mutually exclusive like the touch counts above
  const queriesByLabel = { Included: [], Excluded: [] };
  queries.forEach((q) => {
    included += q.included_concepts.length;
    excluded += q.excluded_concepts.length;
    if (q.included_concepts.length > 0) queriesByLabel.Included.push(q);
    if (q.excluded_concepts.length > 0) queriesByLabel.Excluded.push(q);
  });

  const total = included + excluded;

  const option = {
    title: { text: 'Concept Touches: Included vs Excluded', textStyle: { color: colors.textOnPage } },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: {
      bottom: 0,
      textStyle: { color: colors.textOnPage }
    },
    series: [
      {
        type: 'pie',
        radius: ['30%', '58%'],
        center: ['50%', '58%'],
        avoidLabelOverlap: true,
        label: { color: colors.textOnPage, fontSize: 12 },
        data: [
          { name: 'Included', value: included, itemStyle: { color: colors.categorical[0] } },
          { name: 'Excluded', value: excluded, itemStyle: { color: colors.categorical[7] } }
        ]
      }
    ]
  };

  if (total === 0) {
    return <p style={{ color: colors.textOnPage, fontSize: '0.85rem' }}>No concepts used in this range.</p>;
  }

  return (
    <div>
      <Chart
        option={option}
        style={{ height: '360px', width: '100%' }}
        onEvents={{ click: (params) => setSelectedSplit(params.name) }}
      />

      {selectedSplit && (
        <div style={{ marginTop: '1rem' }}>
          <QueryLinkList queries={queriesByLabel[selectedSplit]} title={selectedSplit} />
        </div>
      )}
    </div>
  );
}

export default InclusionExclusionSplit;
