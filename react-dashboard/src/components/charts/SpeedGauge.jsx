import ReactECharts from 'echarts-for-react';
import { useTheme } from '../../theme';

// round percent gauge, based on the ECharts "custom gauge" example.
// The ring is filled with a theme color instead of the example's image
const MAX = 100;

function toPoint(params, radius, radian) {
  return [
    Math.cos(radian) * radius + params.coordSys.cx,
    -Math.sin(radian) * radius + params.coordSys.cy
  ];
}

function pointerPoints(params, radian) {
  const r = params.coordSys.r;
  return [
    toPoint(params, r, radian),
    toPoint(params, r, radian + Math.PI * 0.03),
    toPoint(params, r * 0.2, radian)
  ];
}

function SpeedGauge({ percent, color, height = 150 }) {
  const { colors } = useTheme();

  function renderItem(params, api) {
    const value = api.value(1);
    const endRadian = api.coord([api.value(0), value])[3];
    const { cx, cy, r } = params.coordSys;

    return {
      type: 'group',
      children: [
        // empty track under the colored ring
        {
          type: 'ring',
          shape: { cx, cy, r, r0: r * 0.85 },
          style: { fill: `${color}22` }
        },
        {
          type: 'sector',
          shape: {
            cx,
            cy,
            r,
            r0: r * 0.85,
            startAngle: 0,
            endAngle: -endRadian,
            transition: 'endAngle',
            enterFrom: { endAngle: 0 }
          },
          style: { fill: color }
        },
        {
          type: 'polygon',
          shape: { points: pointerPoints(params, endRadian) },
          extra: { endRadian, transition: 'endRadian', enterFrom: { endRadian: 0 } },
          during: (apiDuring) => apiDuring.setShape('points', pointerPoints(params, apiDuring.getExtra('endRadian'))),
          style: { fill: color }
        },
        {
          type: 'circle',
          shape: { cx, cy, r: r * 0.7 },
          style: { fill: colors.chartSurface, shadowBlur: 20, shadowColor: `${color}55` }
        },
        {
          type: 'text',
          extra: { value, transition: 'value', enterFrom: { value: 0 } },
          style: {
            text: `${Math.round(value)}%`,
            fontSize: r * 0.35,
            fontWeight: 700,
            x: cx,
            y: cy,
            fill: color,
            align: 'center',
            verticalAlign: 'middle'
          },
          during: (apiDuring) => apiDuring.setStyle('text', `${Math.round(apiDuring.getExtra('value'))}%`)
        }
      ]
    };
  }

  const option = {
    animationDuration: 1000,
    animationDurationUpdate: 1000,
    animationEasingUpdate: 'quarticInOut',
    dataset: { source: [[1, percent]] },
    angleAxis: { type: 'value', startAngle: 0, show: false, min: 0, max: MAX },
    radiusAxis: { type: 'value', show: false },
    polar: { radius: '90%' },
    series: [{ type: 'custom', coordinateSystem: 'polar', renderItem }]
  };

  return <ReactECharts option={option} style={{ height: `${height}px`, width: '100%' }} />;
}

export default SpeedGauge;
