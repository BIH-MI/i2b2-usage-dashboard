import { useTheme } from '../../theme';
import ChartCard from '../ChartCard';
import SectionTitle from './SectionTitle';
import PanelBox from './PanelBox';

// the query as built in i2b2: one panel per line, AND between them
function StructureSection({ panels }) {
  const { colors } = useTheme();
  const included = panels.filter((p) => !p.excluded).length;
  const excluded = panels.length - included;
  const panelWord = panels.length === 1 ? 'panel' : 'panels';

  return (
    <ChartCard info="The query as built in i2b2. Panels are combined with AND, concepts inside one panel with OR. A NOT panel excludes patients who match it.">
      <SectionTitle>Structure</SectionTitle>
      <div style={{ color: `${colors.textOnPage}99`, fontSize: '0.85rem', marginBottom: '1rem' }}>
        {panels.length} {panelWord} · {included} included · {excluded} excluded
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.6rem' }}>
        {panels.map((panel, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0, width: '100%' }}>
            {/* first panel gets a hidden AND so all panels start at the same left edge */}
            <span
              style={{
                visibility: i > 0 ? 'visible' : 'hidden',
                color: colors.textOnPage,
                background: `${colors.textOnPage}14`,
                borderRadius: '999px',
                padding: '0.15rem 0.55rem',
                fontWeight: 700,
                fontSize: '0.65rem',
                letterSpacing: '0.08em'
              }}
            >
              AND
            </span>
            <PanelBox panel={panel} index={i} />
          </div>
        ))}
      </div>
    </ChartCard>
  );
}

export default StructureSection;
