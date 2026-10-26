import React, { useState } from 'react';
import { SIHProblemConfig } from '../data/sihDatasets';
import { EcosystemAnalysisResult, ModelId } from '../engine/dataScienceEngine';

interface OverviewTabProps {
  problem: SIHProblemConfig;
  analysis: EcosystemAnalysisResult;
  selectedModel: ModelId;
  onSelectModel: (m: ModelId) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  problem,
  analysis,
  selectedModel,
  onSelectModel,
}) => {
  const [corrPair, setCorrPair] = useState<{ row: number; col: number }>({ row: 0, col: 10 });
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [windowSize, setWindowSize] = useState<60 | 100 | 160>(100);

  const activeModel = analysis.models[selectedModel];
  const testSlice = activeModel.testPredictions.slice(0, windowSize);

  const svgW = 860;
  const svgH = 320;
  const padL = 62;
  const padR = 24;
  const padT = 18;
  const mainH = 200;
  const resTop = 240;
  const resH = 52;
  const plotW = svgW - padL - padR;

  const allY = testSlice.flatMap((d) => [d.actual, d.predicted]);
  const minY = Math.min(...allY, problem.advisoryThreshold * 0.65) * 0.9;
  const maxY = Math.max(...allY, problem.criticalThreshold * 1.08) * 1.05;
  const maxAbsRes = Math.max(10, ...testSlice.map((d) => Math.abs(d.residual)));

  const scaleX = (i: number) => padL + (i / Math.max(1, testSlice.length - 1)) * plotW;
  const scaleY = (val: number) =>
    padT + mainH - ((val - minY) / Math.max(1, maxY - minY)) * mainH;

  const actualPath = testSlice
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(d.actual).toFixed(1)}`)
    .join(' ');

  const predPath = testSlice
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(d.predicted).toFixed(1)}`)
    .join(' ');

  const bandDelta = activeModel.rmse * 1.65;
  const upperBandPts = testSlice.map(
    (d, i) => `${scaleX(i).toFixed(1)},${scaleY(Math.min(maxY, d.predicted + bandDelta)).toFixed(1)}`
  );
  const lowerBandPts = [...testSlice]
    .reverse()
    .map((d, revI) => {
      const i = testSlice.length - 1 - revI;
      return `${scaleX(i).toFixed(1)},${scaleY(Math.max(minY, d.predicted - bandDelta)).toFixed(1)}`;
    });
  const confidencePolygon = [...upperBandPts, ...lowerBandPts].join(' ');

  const critY = scaleY(problem.criticalThreshold);
  const advY = scaleY(problem.advisoryThreshold);

  const getColValues = (colIdx: number): number[] => {
    if (colIdx === 10) return problem.records.slice(0, 260).map((r) => r.target);
    const key = problem.features[colIdx].key;
    return problem.records.slice(0, 260).map((r) => r[key]);
  };

  const scatterX = getColValues(corrPair.row);
  const scatterY = getColValues(corrPair.col);
  const xMinS = Math.min(...scatterX);
  const xMaxS = Math.max(...scatterX);
  const yMinS = Math.min(...scatterY);
  const yMaxS = Math.max(...scatterY);

  const nS = scatterX.length;
  const meanXS = scatterX.reduce((a, b) => a + b, 0) / nS;
  const meanYS = scatterY.reduce((a, b) => a + b, 0) / nS;
  const numSlope = scatterX.reduce((acc, x, i) => acc + (x - meanXS) * (scatterY[i] - meanYS), 0);
  const denSlope = scatterX.reduce((acc, x) => acc + (x - meanXS) * (x - meanXS), 0) || 1;
  const olsSlope = numSlope / denSlope;
  const olsIntercept = meanYS - olsSlope * meanXS;

  return (
    <div className="space-y-6">
      {/* KPI Metric Strip */}
      <section className="bg-white border border-slate-200 rounded-lg divide-y md:divide-y-0 md:divide-x divide-slate-200 grid grid-cols-2 md:grid-cols-6">
        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">Dataset Sample Size (N)</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-slate-900">
            {analysis.nTotal.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Train: {analysis.nTrain} · Test: {analysis.nTest}
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">Nash-Sutcliffe (NSE)</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-blue-700">
            {activeModel.nse.toFixed(4)}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            KGE: {activeModel.kge.toFixed(4)} · {activeModel.shortName}
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">Explained Variance (R²)</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-slate-900">
            {(activeModel.r2 * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Adj R²: {(activeModel.adjR2 * 100).toFixed(2)}%
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">Test RMSE ({problem.targetUnit})</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-slate-900">
            {activeModel.rmse.toFixed(1)}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            MAE: {activeModel.mae.toFixed(1)} · MAPE: {activeModel.mape.toFixed(2)}%
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">Classification ROC-AUC</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-emerald-700">
            {analysis.classificationEval.rocAuc.toFixed(4)}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            PR-AUC: {analysis.classificationEval.prAuc.toFixed(3)} · F1:{' '}
            {analysis.classificationEval.f1Score.toFixed(3)}
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-medium text-slate-500">100-Yr Gumbel Peak</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-amber-700">
            {analysis.gumbel.returnPeriods.find((r) => r.returnPeriodYears === 100)?.expectedPeakValue.toFixed(0)}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            μ={analysis.gumbel.muLocation.toFixed(0)} · α={analysis.gumbel.alphaScale.toFixed(0)}
          </div>
        </div>
      </section>

      {/* Section 1: Synchronized Dual-Pane Forecast Hydrograph & Residual Impulse Stems */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              01. Out-of-Sample Flood Discharge Hydrograph, 95% Prediction Interval &amp; Residual Impulse Stems
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {([60, 100, 160] as const).map((w) => (
                <button
                  key={w}
                  onClick={() => setWindowSize(w)}
                  className={`px-2.5 py-1 text-xs font-mono font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    windowSize === w
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {w} Pts
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {(['gbdt', 'ridge', 'elastic', 'rf', 'knn'] as const).map((mKey) => (
                <button
                  key={mKey}
                  onClick={() => onSelectModel(mKey)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    selectedModel === mKey
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {analysis.models[mKey].shortName}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-9">
            <svg
              viewBox={`0 0 ${svgW} ${svgH}`}
              className="w-full h-auto select-none bg-slate-50/70 rounded border border-slate-200"
            >
              {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                const yVal = minY + frac * (maxY - minY);
                const yPos = scaleY(yVal);
                return (
                  <g key={frac}>
                    <line
                      x1={padL}
                      y1={yPos}
                      x2={svgW - padR}
                      y2={yPos}
                      stroke="#e2e8f0"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padL - 8}
                      y={yPos + 4}
                      textAnchor="end"
                      className="text-[10px] fill-slate-500 font-mono"
                    >
                      {yVal.toFixed(0)}
                    </text>
                  </g>
                );
              })}

              {advY >= padT && advY <= padT + mainH && (
                <line
                  x1={padL}
                  y1={advY}
                  x2={svgW - padR}
                  y2={advY}
                  stroke="#d97706"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
              )}

              {critY >= padT && critY <= padT + mainH && (
                <g>
                  <line
                    x1={padL}
                    y1={critY}
                    x2={svgW - padR}
                    y2={critY}
                    stroke="#dc2626"
                    strokeWidth="1.25"
                    strokeDasharray="5 4"
                  />
                  <text
                    x={svgW - padR - 6}
                    y={critY - 5}
                    textAnchor="end"
                    className="text-[10px] font-mono fill-red-600 font-semibold"
                  >
                    Danger Limit ({problem.criticalThreshold} {problem.targetUnit})
                  </text>
                </g>
              )}

              <polygon points={confidencePolygon} fill="rgba(37, 99, 235, 0.13)" />
              <path d={actualPath} fill="none" stroke="#0f172a" strokeWidth="1.5" />
              <path d={predPath} fill="none" stroke="#2563eb" strokeWidth="1.8" />

              <line
                x1={padL}
                y1={resTop - 8}
                x2={svgW - padR}
                y2={resTop - 8}
                stroke="#cbd5e1"
              />
              <text
                x={padL - 8}
                y={resTop + resH / 2 + 3}
                textAnchor="end"
                className="text-[9px] font-mono fill-slate-500"
              >
                eᵢ = y - ŷ
              </text>
              <line
                x1={padL}
                y1={resTop + resH / 2}
                x2={svgW - padR}
                y2={resTop + resH / 2}
                stroke="#94a3b8"
                strokeDasharray="2 2"
              />

              {testSlice.map((d, i) => {
                const cx = scaleX(i);
                const cyActual = scaleY(d.actual);
                const cyPred = scaleY(d.predicted);
                const isHovered = hoveredIdx === i;
                const resZeroY = resTop + resH / 2;
                const resTipY = resZeroY - (d.residual / maxAbsRes) * (resH / 2 - 4);

                return (
                  <g
                    key={d.id}
                    onMouseEnter={() => setHoveredIdx(i)}
                    onMouseLeave={() => setHoveredIdx(null)}
                    className="cursor-pointer"
                  >
                    <rect
                      x={cx - plotW / testSlice.length / 2}
                      y={padT}
                      width={plotW / testSlice.length}
                      height={svgH - padT}
                      fill="transparent"
                    />
                    {isHovered && (
                      <line
                        x1={cx}
                        y1={padT}
                        x2={cx}
                        y2={resTop + resH}
                        stroke="#64748b"
                        strokeWidth="1"
                      />
                    )}
                    <line
                      x1={cx}
                      y1={resZeroY}
                      x2={cx}
                      y2={resTipY}
                      stroke={d.residual >= 0 ? '#2563eb' : '#d97706'}
                      strokeWidth={isHovered ? '2.5' : '1.3'}
                    />
                    <circle cx={cx} cy={cyActual} r={isHovered ? 4 : 1.7} fill="#0f172a" />
                    <circle cx={cx} cy={cyPred} r={isHovered ? 4 : 1.7} fill="#2563eb" />
                  </g>
                );
              })}

              {[0, Math.floor(testSlice.length * 0.25), Math.floor(testSlice.length * 0.5), Math.floor(testSlice.length * 0.75), testSlice.length - 1].map((idx) => {
                const item = testSlice[idx];
                if (!item) return null;
                return (
                  <text
                    key={item.id}
                    x={scaleX(idx)}
                    y={svgH - 6}
                    textAnchor="middle"
                    className="text-[10px] fill-slate-500 font-mono"
                  >
                    {item.id}
                  </text>
                );
              })}
            </svg>

            <div className="mt-2.5 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-600">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-slate-900 inline-block" />
                  Observed Q_peak
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-blue-600 inline-block" />
                  {activeModel.shortName} Predicted ŷ
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-2 bg-blue-600/15 border border-blue-600/30 inline-block" />
                  95% CI Band
                </span>
              </div>

              {hoveredIdx !== null && testSlice[hoveredIdx] ? (
                <div className="font-mono text-xs text-slate-900 bg-slate-100 px-2.5 py-0.5 rounded border border-slate-200">
                  {testSlice[hoveredIdx].id} ({testSlice[hoveredIdx].station}) · Obs:{' '}
                  <strong>{testSlice[hoveredIdx].actual}</strong> · Pred:{' '}
                  <strong className="text-blue-700">{testSlice[hoveredIdx].predicted}</strong> · eᵢ ={' '}
                  {testSlice[hoveredIdx].residual > 0 ? '+' : ''}
                  {testSlice[hoveredIdx].residual} m³/s
                </div>
              ) : (
                <div className="text-xs text-slate-400 font-mono">
                  Hover hydrograph points for exact telemetry readout
                </div>
              )}
            </div>
          </div>

          {/* 10-Feature Relative Importance Bar List */}
          <div className="lg:col-span-3 space-y-2 border-t lg:border-t-0 lg:border-l border-slate-200 pt-4 lg:pt-0 lg:pl-5">
            <div className="text-xs font-semibold text-slate-900">
              10-Feature Attribution ({activeModel.shortName})
            </div>
            <div className="space-y-1.5">
              {problem.features.map((feat, idx) => {
                const imp = activeModel.featureImportances[idx] ?? 0;
                return (
                  <div key={feat.key}>
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-slate-700 truncate">
                        {feat.symbol} · {feat.label}
                      </span>
                      <span className="text-slate-900 font-semibold">{imp.toFixed(1)}%</span>
                    </div>
                    <div className="mt-0.5 w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-blue-600 h-full rounded-full"
                        style={{ width: `${Math.min(100, imp * 2.2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 10-Feature Parallel Coordinates & 10-Feature Quantile Box-Whisker Plot */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              02. 10-Feature Parallel Coordinates (Z-Score Regime Centroids)
            </h2>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              {analysis.parallelProfiles.map((p) => (
                <span key={p.regime} className="flex items-center gap-1">
                  <span
                    className={`w-2 h-2 rounded-xs inline-block ${
                      p.regime === 'Severe'
                        ? 'bg-red-600'
                        : p.regime === 'Advisory'
                          ? 'bg-amber-600'
                          : 'bg-emerald-600'
                    }`}
                  />
                  {p.regime} ({p.count})
                </span>
              ))}
            </div>
          </div>

          <div className="mt-3">
            <svg viewBox="0 0 480 230" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              {[
                { z: 1.5, label: '+1.5σ' },
                { z: 0, label: '0.0σ' },
                { z: -1.5, label: '-1.5σ' },
              ].map((ref) => {
                const y = 110 - (ref.z / 2.0) * 78;
                return (
                  <g key={ref.z}>
                    <line
                      x1="42"
                      y1={y}
                      x2="458"
                      y2={y}
                      stroke={ref.z === 0 ? '#94a3b8' : '#e2e8f0'}
                      strokeDasharray={ref.z === 0 ? '4 3' : '2 2'}
                    />
                    <text x="38" y={y + 3} textAnchor="end" className="text-[9px] font-mono fill-slate-500">
                      {ref.label}
                    </text>
                  </g>
                );
              })}

              {problem.features.map((f, j) => {
                const x = 48 + (j / 9) * 404;
                return (
                  <g key={f.key}>
                    <line x1={x} y1="22" x2={x} y2="194" stroke="#cbd5e1" />
                    <text x={x} y="212" textAnchor="middle" className="text-[9px] font-mono font-semibold fill-slate-700">
                      {f.symbol}
                    </text>
                  </g>
                );
              })}

              {analysis.parallelProfiles.map((prof) => {
                const color =
                  prof.regime === 'Severe'
                    ? '#dc2626'
                    : prof.regime === 'Advisory'
                      ? '#d97706'
                      : '#059669';
                const pts = prof.zMeans
                  .map((z, j) => {
                    const x = 48 + (j / 9) * 404;
                    const clampedZ = Math.max(-2.0, Math.min(2.0, z));
                    const y = 110 - (clampedZ / 2.0) * 78;
                    return `${j === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                  })
                  .join(' ');

                return (
                  <g key={prof.regime}>
                    <path d={pts} fill="none" stroke={color} strokeWidth="2" />
                    {prof.zMeans.map((z, j) => {
                      const x = 48 + (j / 9) * 404;
                      const clampedZ = Math.max(-2.0, Math.min(2.0, z));
                      const y = 110 - (clampedZ / 2.0) * 78;
                      return (
                        <circle key={j} cx={x} cy={y} r="3" fill={color}>
                          <title>
                            {prof.regime} · {problem.features[j].symbol}: z = {z >= 0 ? '+' : ''}
                            {z}
                          </title>
                        </circle>
                      );
                    })}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* 10-Feature Normalized Box-and-Whisker Quantile Spread Plot */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              03. 10-Feature Box-and-Whisker Quantile Spread (Q₁₀–Q₉₀ &amp; IQR)
            </h2>
            <span className="text-[11px] font-mono text-slate-500">
              Box = IQR · Diamond = Mean μ
            </span>
          </div>

          <div className="mt-3 space-y-2">
            {analysis.featureStats.map((st) => {
              const span = Math.max(1e-6, st.max - st.min);
              const p10 = ((st.q10 - st.min) / span) * 100;
              const p25 = ((st.q25 - st.min) / span) * 100;
              const p50 = ((st.median - st.min) / span) * 100;
              const p75 = ((st.q75 - st.min) / span) * 100;
              const p90 = ((st.q90 - st.min) / span) * 100;
              const pMean = ((st.mean - st.min) / span) * 100;

              return (
                <div key={st.key} className="grid grid-cols-12 items-center gap-2 text-xs font-mono">
                  <div className="col-span-3 truncate font-semibold text-slate-800" title={st.label}>
                    {st.symbol} <span className="text-[10px] font-normal text-slate-500">({st.unit})</span>
                  </div>
                  <div className="col-span-6 relative h-4 bg-slate-100 rounded border border-slate-200">
                    <div
                      className="absolute top-1/2 -translate-y-1/2 h-0.5 bg-slate-500"
                      style={{ left: `${p10}%`, width: `${Math.max(1, p90 - p10)}%` }}
                    />
                    <div
                      className="absolute top-0.5 bottom-0.5 bg-blue-600/20 border border-blue-600 rounded-xs"
                      style={{ left: `${p25}%`, width: `${Math.max(2, p75 - p25)}%` }}
                    />
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-blue-700"
                      style={{ left: `${p50}%` }}
                    />
                    <div
                      className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-amber-500 rotate-45"
                      style={{ left: `calc(${pMean}% - 3px)` }}
                    />
                  </div>
                  <div className="col-span-3 text-right text-[11px] text-slate-600 tabular-nums">
                    {st.q25.toFixed(0)}–{st.q75.toFixed(0)} (μ={st.mean.toFixed(0)})
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Section 3: Interactive 11x11 Pearson Correlation Matrix + Bivariate OLS Scatter Inspector */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              04. 11×11 Pearson Correlation Matrix (R = (1/(n-1)) ZᵀZ)
            </h2>
            <span className="text-xs text-slate-500">Click any cell to plot OLS fit</span>
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full border-collapse text-[11px] font-mono tabular-nums">
              <thead>
                <tr>
                  <th className="p-1.5 text-left text-slate-500 font-medium border-b border-slate-200">
                    Var
                  </th>
                  {analysis.correlationMatrix.symbols.map((sym, idx) => (
                    <th
                      key={sym}
                      className="p-1.5 text-right text-slate-700 font-semibold border-b border-slate-200"
                      title={analysis.correlationMatrix.labels[idx]}
                    >
                      {sym}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {analysis.correlationMatrix.matrix.map((row, rIdx) => (
                  <tr key={analysis.correlationMatrix.symbols[rIdx]} className="border-b border-slate-100">
                    <td className="p-1.5 font-semibold text-slate-800 whitespace-nowrap">
                      {analysis.correlationMatrix.symbols[rIdx]}
                    </td>
                    {row.map((val, cIdx) => {
                      const isSelected = corrPair.row === rIdx && corrPair.col === cIdx;
                      const bgStyle =
                        rIdx === cIdx
                          ? 'bg-slate-100 text-slate-900'
                          : val > 0.6
                            ? 'bg-blue-600 text-white'
                            : val > 0.3
                              ? 'bg-blue-100 text-blue-950'
                              : val > 0.1
                                ? 'bg-blue-50 text-slate-800'
                                : val < -0.3
                                  ? 'bg-amber-200 text-amber-950'
                                  : val < -0.1
                                    ? 'bg-amber-50 text-slate-800'
                                    : 'bg-white text-slate-600';

                      return (
                        <td
                          key={cIdx}
                          onClick={() => setCorrPair({ row: rIdx, col: cIdx })}
                          className={`p-1.5 text-right cursor-pointer transition-opacity hover:opacity-80 ${bgStyle} ${
                            isSelected ? 'ring-2 ring-slate-900 ring-inset font-bold' : ''
                          }`}
                        >
                          {val.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Bivariate Scatter Plot & OLS Regression Inspector */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-900">
                Bivariate OLS Regression Fit
              </h2>
              <span className="text-xs font-mono text-blue-700 font-semibold">
                {analysis.correlationMatrix.symbols[corrPair.row]} vs{' '}
                {analysis.correlationMatrix.symbols[corrPair.col]}
              </span>
            </div>

            <div className="mt-3">
              <svg viewBox="0 0 380 215" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
                <line x1="46" y1="16" x2="46" y2="188" stroke="#cbd5e1" />
                <line x1="46" y1="188" x2="362" y2="188" stroke="#cbd5e1" />

                {scatterX.map((xv, i) => {
                  const yv = scatterY[i];
                  const cx = 46 + ((xv - xMinS) / Math.max(1e-6, xMaxS - xMinS)) * 310;
                  const cy = 188 - ((yv - yMinS) / Math.max(1e-6, yMaxS - yMinS)) * 165;
                  return (
                    <circle
                      key={i}
                      cx={cx}
                      cy={cy}
                      r="2.3"
                      fill="#2563eb"
                      fillOpacity="0.5"
                    />
                  );
                })}

                {(() => {
                  const yAtMin = olsIntercept + olsSlope * xMinS;
                  const yAtMax = olsIntercept + olsSlope * xMaxS;
                  const y1 = 188 - ((yAtMin - yMinS) / Math.max(1e-6, yMaxS - yMinS)) * 165;
                  const y2 = 188 - ((yAtMax - yMinS) / Math.max(1e-6, yMaxS - yMinS)) * 165;
                  return (
                    <line
                      x1="46"
                      y1={Math.max(16, Math.min(188, y1))}
                      x2="356"
                      y2={Math.max(16, Math.min(188, y2))}
                      stroke="#dc2626"
                      strokeWidth="2"
                    />
                  );
                })()}

                <text x="200" y="208" textAnchor="middle" className="text-[10px] font-mono fill-slate-600">
                  {analysis.correlationMatrix.labels[corrPair.row]} ({xMinS.toFixed(1)} – {xMaxS.toFixed(1)})
                </text>
              </svg>
            </div>
          </div>

          <div className="mt-3 pt-2.5 border-t border-slate-200 flex items-center justify-between text-xs font-mono text-slate-700">
            <span>
              Pearson r ={' '}
              <strong>
                {analysis.correlationMatrix.matrix[corrPair.row][corrPair.col].toFixed(4)}
              </strong>
            </span>
            <span>
              OLS: ŷ = {olsSlope.toFixed(2)}x {olsIntercept >= 0 ? '+' : ''}{' '}
              {olsIntercept.toFixed(1)}
            </span>
          </div>
        </div>
      </section>

      {/* Section 4: 10D PCA Biplot & 12-River Basin Telemetry Stratification Table */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              05. 10D PCA Biplot &amp; Loading Vectors
            </h2>
            <span className="text-xs font-mono text-slate-500">
              PC1 ({analysis.pca.explainedVariancePct[0]}%) + PC2 ({analysis.pca.explainedVariancePct[1]}%)
            </span>
          </div>

          <div className="mt-3">
            <svg viewBox="0 0 380 220" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              <line x1="190" y1="12" x2="190" y2="208" stroke="#cbd5e1" strokeDasharray="3 3" />
              <line x1="20" y1="110" x2="360" y2="110" stroke="#cbd5e1" strokeDasharray="3 3" />
              {analysis.pca.projectedPoints.map((pt) => {
                const cx = Math.max(24, Math.min(356, 190 + pt.pc1 * 36));
                const cy = Math.max(16, Math.min(204, 110 - pt.pc2 * 36));
                const color =
                  pt.riskClass === 'Severe'
                    ? '#dc2626'
                    : pt.riskClass === 'Advisory'
                      ? '#d97706'
                      : '#2563eb';
                return (
                  <circle
                    key={pt.id}
                    cx={cx}
                    cy={cy}
                    r="2.2"
                    fill={color}
                    fillOpacity="0.52"
                  />
                );
              })}

              {problem.features.slice(0, 6).map((f, j) => {
                const vx = (analysis.pca.loadings[0]?.[j] ?? 0) * 115;
                const vy = (analysis.pca.loadings[1]?.[j] ?? 0) * 115;
                const x2 = 190 + vx;
                const y2 = 110 - vy;
                return (
                  <g key={f.key}>
                    <line x1="190" y1="110" x2={x2} y2={y2} stroke="#0f172a" strokeWidth="1.5" />
                    <circle cx={x2} cy={y2} r="2.2" fill="#0f172a" />
                    <text
                      x={x2 + (vx >= 0 ? 4 : -4)}
                      y={y2 + (vy >= 0 ? -3 : 7)}
                      textAnchor={vx >= 0 ? 'start' : 'end'}
                      className="text-[8.5px] font-mono font-bold fill-slate-900"
                    >
                      {f.symbol}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="mt-3 grid grid-cols-5 gap-1.5 font-mono text-[11px]">
            {analysis.pca.eigenvalues.slice(0, 5).map((ev, k) => (
              <div key={k} className="p-2 bg-slate-50 border border-slate-200 rounded text-center">
                <div className="text-slate-500">PC{k + 1}</div>
                <div className="font-bold text-slate-900">{ev.toFixed(2)}</div>
                <div className="text-[10px] text-blue-700">{analysis.pca.explainedVariancePct[k]}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* 12-River Basin Station Stratification Table */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              06. 12 National River Basin Gauge &amp; Dam Stratification
            </h2>
            <span className="text-xs font-mono text-slate-500">
              Total N = {analysis.nTotal.toLocaleString()}
            </span>
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-medium">
                  <th className="py-2 pr-3">River Basin &amp; Gauge Station</th>
                  <th className="py-2 px-2 text-right font-mono">n</th>
                  <th className="py-2 px-2 text-right font-mono">Mean R_24</th>
                  <th className="py-2 px-2 text-right font-mono">Water Level</th>
                  <th className="py-2 px-2 text-right font-mono">Mean Q_peak</th>
                  <th className="py-2 px-2 text-right font-mono">Max Q_peak</th>
                  <th className="py-2 pl-2 text-right font-mono">Severe Floods</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                {analysis.stationSummaries.map((st) => (
                  <tr key={st.station} className="hover:bg-slate-50">
                    <td className="py-2 pr-3 font-sans font-medium text-slate-900">
                      {st.station}
                    </td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.count}</td>
                    <td className="py-2 px-2 text-right text-slate-700">{st.meanX1} mm</td>
                    <td className="py-2 px-2 text-right text-slate-700">{st.meanWaterLevel} m</td>
                    <td className="py-2 px-2 text-right font-semibold text-slate-900">
                      {st.meanTarget} m³/s
                    </td>
                    <td className="py-2 px-2 text-right text-blue-700 font-semibold">
                      {st.maxTarget} m³/s
                    </td>
                    <td className="py-2 pl-2 text-right text-slate-800">
                      {st.severeCount} ({st.severeRatePct}%)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
};
