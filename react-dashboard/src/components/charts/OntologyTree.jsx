import Chart from '../Chart';
import { useTheme } from '../../theme';

// concept paths as a nested tree: ontology, then folders, then concept
function buildTree(concepts) {
  const root = new Map();
  concepts.forEach((path) => {
    let level = root;
    path
      .split('\\') 
      .filter(Boolean)
      .filter((segment, i) => i === 0 || segment.toLowerCase() !== 'i2b2') 
      .forEach((segment) => {
        const name = segment.replace(/^i2b2_/, '');
        if (!level.has(name)) level.set(name, new Map());
        level = level.get(name);
      });
  });

  const toNodes = (level) =>
    [...level.entries()].map(([name, children]) => {
      if (children.size === 0) return { name, count: 1 };
      // a folder with only one sub folder is merged into it, so the tree has no single-step branches
      let label = name;
      let current = children;
      while (current.size === 1) {
        const [[childName, grandChildren]] = [...current.entries()];
        if (grandChildren.size === 0) break;
        label = `${label} › ${childName}`;
        current = grandChildren;
      }
      const kids = toNodes(current);
      return { name: label, count: kids.reduce((sum, k) => sum + k.count, 0), children: kids };
    });
  return toNodes(root);
}

// every node of one ontology gets that ontology's color
function paint(node, color) {
  return {
    ...node,
    itemStyle: { color: node.children ? color : `${color}55`, borderColor: color },
    children: node.children?.map((child) => paint(child, color))
  };
}

// collapsible tree of the query's concepts. Starts at the ontologies, a click on a node opens
// or closes its branch. Scroll to zoom, drag to move
function OntologyTree({ concepts }) {
  const { colors } = useTheme();
  const unique = [...new Set(concepts)];
  if (unique.length === 0) return null;

  const ontologies = buildTree(unique).map((node, i) => paint(node, colors.categorical[i % colors.categorical.length]));
  const data = {
    name: 'Query',
    count: unique.length,
    itemStyle: { color: colors.textOnPage, borderColor: colors.textOnPage },
    children: ontologies
  };

  const option = {
    title: { text: 'Ontology Tree', textStyle: { color: colors.textOnPage } },
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.name}<br/>${p.data.count} concept${p.data.count === 1 ? '' : 's'}`,
      extraCssText: 'max-width: 340px; white-space: normal;'
    },
    series: [
      {
        type: 'tree',
        data: [data],
        top: 50,
        bottom: 20,
        left: 70,
        right: 220,
        initialTreeDepth: 1, // only the ontologies are open at the start
        expandAndCollapse: true,
        roam: true,
        symbol: 'circle',
        symbolSize: 10,
        lineStyle: { color: colors.gridLine, width: 1.5, curveness: 0.5 },
        label: {
          position: 'left',
          verticalAlign: 'middle',
          align: 'right',
          fontSize: 12,
          color: colors.textOnPage,
          // folders show how many concepts are inside
          formatter: (p) => (p.data.children ? `${p.name} (${p.data.count})` : p.name)
        },
        leaves: { label: { position: 'right', align: 'left' } },
        emphasis: { focus: 'descendant' },
        animationDuration: 400,
        animationDurationUpdate: 500
      }
    ]
  };

  return <Chart option={option} style={{ height: '460px', width: '100%' }} />;
}

export default OntologyTree;
