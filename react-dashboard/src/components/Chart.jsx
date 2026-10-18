import { useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { Download } from 'lucide-react';
import { useTheme } from '../theme';
import { useUserFilter } from '../context/UserFilterContext';

// gives every x/y axis the theme's grid line color, unless the chart set its own
function withGridColor(axis, color) {
  if (!axis) return axis;
  const one = (a) => ({ ...a, splitLine: { ...a.splitLine, lineStyle: { color, ...a.splitLine?.lineStyle } } });
  return Array.isArray(axis) ? axis.map(one) : one(axis);
}

// copy of the option with every username replaced by its alias, object keys too (legend.selected uses names as keys).
// only exact matches change. functions and class objects (e.g. gradients) stay as they are
function replaceNames(value, names) {
  if (typeof value === 'string') return names.get(value) ?? value;
  if (Array.isArray(value)) return value.map((v) => replaceNames(v, names));
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [names.get(k) ?? k, replaceNames(v, names)]));
  }
  return value;
}

// click handlers get the real username back, they use it to load that user's queries
function withRealNames(onEvents, realNames) {
  if (!onEvents) return onEvents;
  return Object.fromEntries(
    Object.entries(onEvents).map(([event, handler]) => [
      event,
      (params) =>
        handler({
          ...params,
          name: replaceNames(params.name, realNames),
          seriesName: replaceNames(params.seriesName, realNames),
          data: replaceNames(params.data, realNames)
        })
    ])
  );
}

// thin wrapper around ReactECharts. Renders its own download icon (not ECharts' built-in
// toolbox) so it sits right next to ChartCard's info icon, same size/color/position system.
function Chart({ option: rawOption, onEvents, ...rest }) {
  const { colors } = useTheme();
  const { hideUsernames, aliases } = useUserFilter();
  const styledOption = {
    ...rawOption,
    ...(rawOption.title && { title: { left: 'center', ...rawOption.title } }), // every chart title centered, unless the chart sets its own position
    xAxis: withGridColor(rawOption.xAxis, colors.gridLine),
    yAxis: withGridColor(rawOption.yAxis, colors.gridLine)
  };
  // with "Usernames hidden" on, the chart and its PNG show User 1, User 2... instead of the names
  const option = hideUsernames ? replaceNames(styledOption, aliases) : styledOption;
  const events = hideUsernames ? withRealNames(onEvents, new Map([...aliases].map(([name, alias]) => [alias, name]))) : onEvents;
  const [instance, setInstance] = useState(null);

  function downloadPng() {
    if (!instance) return;
    const url = instance.getDataURL({ backgroundColor: colors.chartSurface, pixelRatio: 2 });
    const filename = `${(option.title?.text || 'chart').replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.png`;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
  }

  return (
    <>
      <div
        onClick={downloadPng}
        title="Download as PNG"
        // sits left of the info icon, the gap follows the icon size
        style={{ position: 'absolute', top: '0.75rem', right: 'calc(var(--icon-size) + 1.4rem)', zIndex: 20, cursor: 'pointer' }}
      >
        <Download color={`${colors.textOnPage}99`} style={{ display: 'block', width: 'var(--icon-size)', height: 'var(--icon-size)' }} />
      </div>

      <ReactECharts option={option} onEvents={events} onChartReady={setInstance} {...rest} />
    </>
  );
}

export default Chart;
