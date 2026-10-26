import React, { useMemo, useState } from 'react';
import { FEATURE_KEYS, FeatureKey, SIHProblemConfig } from '../data/sihDatasets';
import { EcosystemAnalysisResult } from '../engine/dataScienceEngine';
import { RotateCcw } from 'lucide-react';

interface SimulatorTabProps {
  problem: SIHProblemConfig;
  analysis: EcosystemAnalysisResult;
}

export const SimulatorTab: React.FC<SimulatorTabProps> = ({ problem, analysis }) => {
  const defaultValues = useMemo(() => {
    const obj = {} as Record<FeatureKey, number>;
    problem.features.forEach((f) => {
      obj[f.key] = f.defaultVal;
    });
    return obj;
  }, [problem]);

  const [inputs, setInputs] = useState<Record<FeatureKey, number>>(defaultValues);
  const [pdpFeatureKey, setPdpFeatureKey] = useState<FeatureKey>('x1');

  const simResult = useMemo(() => {
    return analysis.predictSingle(inputs);
  }, [analysis, inputs]);

  const pdpCurve = useMemo(() => {
    return analysis.computePartialDependence(pdpFeatureKey, inputs);
  }, [analysis, pdpFeatureKey, inputs]);

  const pdpFeatureSpec = useMemo(() => {
    return problem.features.find((f) => f.key === pdpFeatureKey) || problem.features[0];
  }, [problem.features, pdpFeatureKey]);

  const applyPreset = (mode: 'normal' | 'advisory' | 'severe') => {
    const next = {} as Record<FeatureKey, number>;
    problem.features.forEach((f) => {
      if (mode === 'normal') {
        next[f.key] = f.defaultVal;
      } else if (mode === 'advisory') {
        next[f.key] = Number((f.defaultVal + (f.max - f.defaultVal) * 0.45).toFixed(2));
      } else {
        next[f.key] = Number((f.defaultVal + (f.max - f.defaultVal) * 0.85).toFixed(2));
      }
    });
    setInputs(next);
  };

  const maxAbsShap = Math.max(
    50,
    ...simResult.shapContributions.map((c) => Math.abs(c.contribution))
  );

  // PDP SVG layout
  const pdpW = 540;
  const pdpH = 220;
  const padL = 54;
  const padR = 20;
  const padT = 16;
  const padB = 32;
  const plotW = pdpW - padL - padR;
  const plotH = pdpH - padT - padB;

  const pdpAllY = pdpCurve.flatMap((pt) => [pt.gbdt, pt.ridge, pt.elastic, pt.rf, pt.ensemble]);
  const pdpMinY = Math.min(...pdpAllY, problem.advisoryThreshold * 0.7) * 0.92;
  const pdpMaxY = Math.max(...pdpAllY, problem.criticalThreshold * 1.05) * 1.05;

  const scaleX = (xVal: number) =>
    padL +
    ((xVal - pdpFeatureSpec.min) / Math.max(1e-6, pdpFeatureSpec.max - pdpFeatureSpec.min)) *
      plotW;
  const scaleY = (yVal: number) =>
    padT + plotH - ((yVal - pdpMinY) / Math.max(1, pdpMaxY - pdpMinY)) * plotH;

  const makePath = (key: 'gbdt' | 'ridge' | 'elastic' | 'rf' | 'ensemble') =>
    pdpCurve
      .map(
        (pt, idx) =>
          `${idx === 0 ? 'M' : 'L'} ${scaleX(pt.xVal).toFixed(1)} ${scaleY(pt[key]).toFixed(1)}`
      )
      .join(' ');

  const currentXCoord = scaleX(inputs[pdpFeatureKey]);

  const isSevere = simResult.ensembleMean >= problem.criticalThreshold;
  const isAdvisory =
    !isSevere && simResult.ensembleMean >= problem.advisoryThreshold;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: 10-Feature Telemetry Input Sliders */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 pb-4 mb-4 border-b border-slate-200">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  10-Feature Telemetry Vector (x₁ … x₁₀)
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Adjust river basin parameters for live 5-model flood discharge &amp; SHAP attribution
                </p>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => applyPreset('normal')}
                  className="px-2.5 py-1 rounded text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors cursor-pointer"
                >
                  Baseflow
                </button>
                <button
                  onClick={() => applyPreset('advisory')}
                  className="px-2.5 py-1 rounded text-xs font-medium bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 transition-colors cursor-pointer"
                >
                  Monsoon Surge
                </button>
                <button
                  onClick={() => applyPreset('severe')}
                  className="px-2.5 py-1 rounded text-xs font-medium bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 transition-colors cursor-pointer"
                >
                  Cloudburst
                </button>
                <button
                  onClick={() => setInputs(defaultValues)}
                  title="Reset parameters"
                  className="p-1.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {problem.features.map((f, idx) => {
                const val = inputs[f.key] ?? f.defaultVal;
                const fStat = analysis.featureStats[idx];
                const zScore = fStat ? (val - fStat.mean) / Math.max(1e-6, fStat.std) : 0;

                return (
                  <div
                    key={f.key}
                    className="p-2.5 bg-slate-50 border border-slate-200 rounded-md space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="font-mono text-[10px] font-semibold text-blue-700 shrink-0">
                          {f.symbol}
                        </span>
                        <span className="font-medium text-slate-800 truncate" title={f.label}>
                          {f.label}
                        </span>
                      </div>
                      <span className="font-mono font-bold text-slate-900 tabular-nums shrink-0 ml-1">
                        {val}{' '}
                        <span className="text-[10px] font-normal text-slate-500">{f.unit}</span>
                      </span>
                    </div>

                    <input
                      type="range"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={val}
                      onChange={(e) =>
                        setInputs((prev) => ({
                          ...prev,
                          [f.key]: parseFloat(e.target.value),
                        }))
                      }
                      className="w-full accent-blue-600 cursor-pointer h-1.5"
                    />

                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 tabular-nums">
                      <span>{f.min}</span>
                      <span
                        className={
                          Math.abs(zScore) > 1.5
                            ? 'text-amber-700 font-semibold'
                            : 'text-slate-500'
                        }
                      >
                        z = {zScore >= 0 ? `+${zScore.toFixed(2)}` : zScore.toFixed(2)}σ
                      </span>
                      <span>{f.max}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
            <span>Training Baseline Mean E[f(X)]</span>
            <span className="font-mono font-semibold text-slate-900 tabular-nums">
              {simResult.baselinePrediction.toLocaleString()} {problem.targetUnit}
            </span>
          </div>
        </div>

        {/* Right Column: 5-Model Readout, SHAP Waterfall & 1D PDP */}
        <div className="lg:col-span-7 space-y-6">
          {/* Top Multi-Model Prediction & Risk Card */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pb-4 border-b border-slate-200">
              <div className="md:col-span-2">
                <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
                  5-Model Ensemble Consensus • {problem.targetLabel} ({problem.targetSymbol})
                </div>
                <div className="flex items-baseline gap-2.5 mt-1">
                  <span className="text-3xl font-bold font-mono text-slate-900 tabular-nums">
                    {simResult.ensembleMean.toLocaleString()}
                  </span>
                  <span className="text-sm font-mono text-blue-700 font-semibold">
                    {problem.targetUnit}
                  </span>
                  <span className="text-xs font-mono text-slate-500 tabular-nums ml-2">
                    (±σ_ens = {simResult.ensembleStd.toLocaleString()} {problem.targetUnit})
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-600 mt-1.5 tabular-nums">
                  95% Prediction Interval:{' '}
                  <strong className="text-slate-900">
                    [{simResult.ci95Low.toLocaleString()}, {simResult.ci95High.toLocaleString()}]{' '}
                    {problem.targetUnit}
                  </strong>
                </div>
              </div>

              <div className="flex flex-col justify-between md:items-end border-t md:border-t-0 pt-3 md:pt-0 border-slate-200">
                <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
                  CWC Flood Alert &amp; Stage
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      isSevere
                        ? 'bg-rose-600'
                        : isAdvisory
                        ? 'bg-amber-500'
                        : 'bg-emerald-600'
                    }`}
                  />
                  <span
                    className={`text-sm font-mono font-bold ${
                      isSevere
                        ? 'text-rose-700'
                        : isAdvisory
                        ? 'text-amber-700'
                        : 'text-emerald-700'
                    }`}
                  >
                    {isSevere
                      ? problem.criticalClassLabel
                      : isAdvisory
                      ? problem.advisoryClassLabel
                      : problem.normalClassLabel}
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-600 mt-1 tabular-nums">
                  Gauge Level:{' '}
                  <strong className="text-slate-900">{simResult.estimatedWaterLevelM} m</strong> • T ≈{' '}
                  <strong className="text-blue-700">{simResult.estimatedReturnPeriod} yr</strong> (P_crit:{' '}
                  {simResult.criticalProbability}%)
                </div>
              </div>
            </div>

            {/* 5 Individual ML Model Outputs */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
                <div className="text-[11px] font-medium text-slate-500">GBDT / XGBoost</div>
                <div className="text-base font-mono font-bold text-blue-700 mt-0.5 tabular-nums">
                  {simResult.gbdt.toLocaleString()}
                </div>
                <div className="text-[10px] font-mono text-slate-400">{problem.targetUnit}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
                <div className="text-[11px] font-medium text-slate-500">Ridge (L2)</div>
                <div className="text-base font-mono font-bold text-indigo-700 mt-0.5 tabular-nums">
                  {simResult.ridge.toLocaleString()}
                </div>
                <div className="text-[10px] font-mono text-slate-400">{problem.targetUnit}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
                <div className="text-[11px] font-medium text-slate-500">ElasticNet (L1/L2)</div>
                <div className="text-base font-mono font-bold text-amber-700 mt-0.5 tabular-nums">
                  {simResult.elastic.toLocaleString()}
                </div>
                <div className="text-[10px] font-mono text-slate-400">{problem.targetUnit}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
                <div className="text-[11px] font-medium text-slate-500">Random Forest</div>
                <div className="text-base font-mono font-bold text-emerald-700 mt-0.5 tabular-nums">
                  {simResult.rf.toLocaleString()}
                </div>
                <div className="text-[10px] font-mono text-slate-400">{problem.targetUnit}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
                <div className="text-[11px] font-medium text-slate-500">Weighted k-NN</div>
                <div className="text-base font-mono font-bold text-rose-700 mt-0.5 tabular-nums">
                  {simResult.knn.toLocaleString()}
                </div>
                <div className="text-[10px] font-mono text-slate-400">{problem.targetUnit}</div>
              </div>
            </div>
          </div>

          {/* 10-Feature Local SHAP Attribution Waterfall */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-200">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  10-Feature Local SHAP Attribution Waterfall — φⱼ(x) = β̂ⱼ · (xⱼ − μⱼ)
                </h3>
                <p className="text-xs text-slate-500">
                  Additive contribution of each hydro-meteorological variable from baseline E[f(X)] ={' '}
                  {simResult.baselinePrediction.toLocaleString()} {problem.targetUnit}
                </p>
              </div>
              <span className="text-xs font-mono text-slate-600 tabular-nums">
                f(x) = {simResult.baselinePrediction.toLocaleString()} + Σφⱼ ={' '}
                <strong className="text-slate-900">{simResult.ridge.toLocaleString()}</strong>
              </span>
            </div>

            <div className="space-y-2">
              {simResult.shapContributions.map((item) => {
                const pct = Math.min(100, (Math.abs(item.contribution) / maxAbsShap) * 100);
                const isPos = item.contribution >= 0;
                return (
                  <div key={item.key} className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-4 flex items-center gap-1.5 truncate">
                      <span className="font-mono text-[10px] font-semibold text-blue-700 w-7 shrink-0">
                        {item.symbol}
                      </span>
                      <span className="text-slate-700 truncate" title={item.label}>
                        {item.label}
                      </span>
                    </div>
                    <div className="col-span-2 text-right font-mono text-[11px] text-slate-500 tabular-nums">
                      {item.value} (μ={item.meanValue})
                    </div>
                    <div className="col-span-4 flex items-center">
                      <div className="w-full h-3 bg-slate-100 rounded-sm overflow-hidden flex">
                        <div className="w-1/2 flex justify-end border-r border-slate-300">
                          {!isPos && (
                            <div
                              className="h-full bg-emerald-500"
                              style={{ width: `${pct}%` }}
                            />
                          )}
                        </div>
                        <div className="w-1/2 flex justify-start">
                          {isPos && (
                            <div
                              className="h-full bg-rose-500"
                              style={{ width: `${pct}%` }}
                            />
                          )}
                        </div>
                      </div>
                    </div>
                    <div
                      className={`col-span-2 text-right font-mono font-semibold tabular-nums ${
                        isPos ? 'text-rose-700' : 'text-emerald-700'
                      }`}
                    >
                      {isPos ? `+${item.contribution}` : item.contribution}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 1D Partial Dependence Plot (PDP) Across Selected Feature */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-200">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  1D Partial Dependence Plot (PDP) — Multi-Model Response Sensitivity
                </h3>
                <p className="text-xs text-slate-500">
                  Marginal effect of varying {pdpFeatureSpec.label} ({pdpFeatureSpec.symbol}) across [
                  {pdpFeatureSpec.min}, {pdpFeatureSpec.max}] {pdpFeatureSpec.unit}
                </p>
              </div>
              <select
                value={pdpFeatureKey}
                onChange={(e) => setPdpFeatureKey(e.target.value as FeatureKey)}
                className="px-2.5 py-1.5 text-xs font-medium bg-slate-50 border border-slate-300 rounded-md text-slate-800"
              >
                {FEATURE_KEYS.map((k, idx) => (
                  <option key={k} value={k}>
                    {problem.features[idx].symbol}: {problem.features[idx].label}
                  </option>
                ))}
              </select>
            </div>

            <svg viewBox={`0 0 ${pdpW} ${pdpH}`} className="w-full h-auto overflow-visible">
              {/* Horizontal reference lines */}
              {[0, 0.5, 1].map((t, i) => {
                const yVal = pdpMinY + t * (pdpMaxY - pdpMinY);
                const yPos = scaleY(yVal);
                return (
                  <g key={i}>
                    <line
                      x1={padL}
                      y1={yPos}
                      x2={pdpW - padR}
                      y2={yPos}
                      stroke="#e2e8f0"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padL - 6}
                      y={yPos + 3}
                      textAnchor="end"
                      className="text-[10px] fill-slate-500 font-mono"
                    >
                      {Math.round(yVal)}
                    </text>
                  </g>
                );
              })}

              {/* Critical threshold line */}
              {problem.criticalThreshold >= pdpMinY && problem.criticalThreshold <= pdpMaxY && (
                <line
                  x1={padL}
                  y1={scaleY(problem.criticalThreshold)}
                  x2={pdpW - padR}
                  y2={scaleY(problem.criticalThreshold)}
                  stroke="#e11d48"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
              )}

              {/* Current selected value vertical line */}
              <line
                x1={currentXCoord}
                y1={padT}
                x2={currentXCoord}
                y2={pdpH - padB}
                stroke="#f59e0b"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />

              {/* Model response curves */}
              <path
                d={makePath('ridge')}
                fill="none"
                stroke="#6366f1"
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
              <path d={makePath('rf')} fill="none" stroke="#10b981" strokeWidth={1.5} />
              <path d={makePath('gbdt')} fill="none" stroke="#0284c7" strokeWidth={2.2} />
              <path d={makePath('ensemble')} fill="none" stroke="#0f172a" strokeWidth={2} />

              {/* X-axis ticks */}
              {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
                const xVal =
                  pdpFeatureSpec.min + t * (pdpFeatureSpec.max - pdpFeatureSpec.min);
                const xPos = scaleX(xVal);
                return (
                  <text
                    key={i}
                    x={xPos}
                    y={pdpH - 10}
                    textAnchor="middle"
                    className="text-[10px] fill-slate-500 font-mono"
                  >
                    {xVal.toFixed(0)} {pdpFeatureSpec.unit}
                  </text>
                );
              })}
            </svg>

            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600 pt-1">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-slate-900 inline-block" /> Ensemble Consensus
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-sky-600 inline-block" /> GBDT
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-indigo-500 inline-block" /> Ridge (L2)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-emerald-500 inline-block" /> Random Forest
                </span>
              </div>
              <span className="font-mono text-amber-700">
                Current {pdpFeatureSpec.symbol} = {inputs[pdpFeatureKey]} {pdpFeatureSpec.unit}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
