import React, { useState } from 'react';
import { SIHProblemConfig } from '../data/sihDatasets';
import { EcosystemAnalysisResult, Hyperparameters, ModelId } from '../engine/dataScienceEngine';

interface ModelStudioTabProps {
  problem: SIHProblemConfig;
  analysis: EcosystemAnalysisResult;
  hyper: Hyperparameters;
  onChangeHyper: (next: Hyperparameters) => void;
  selectedModel: ModelId;
  onSelectModel: (m: ModelId) => void;
}

export const ModelStudioTab: React.FC<ModelStudioTabProps> = ({
  problem,
  analysis,
  hyper,
  onChangeHyper,
  selectedModel,
  onSelectModel,
}) => {
  const [classChartMode, setClassChartMode] = useState<'roc' | 'pr' | 'calibration'>('roc');

  const activeModel = analysis.models[selectedModel];
  const preds = activeModel.testPredictions.slice(0, 220);

  const allVals = preds.flatMap((p) => [p.actual, p.predicted]);
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);

  const maxAbsRes = Math.max(10, ...preds.map((p) => Math.abs(p.residual)));

  const radarAxes = [
    { label: 'R² Fit', angle: -Math.PI / 2 },
    { label: 'NSE Score', angle: -Math.PI / 2 + (2 * Math.PI) / 5 },
    { label: 'KGE Score', angle: -Math.PI / 2 + (4 * Math.PI) / 5 },
    { label: '1 - MAPE', angle: -Math.PI / 2 + (6 * Math.PI) / 5 },
    { label: 'CV Stability', angle: -Math.PI / 2 + (8 * Math.PI) / 5 },
  ];

  const getModelRadarScores = (mKey: ModelId) => {
    const m = analysis.models[mKey];
    return [
      Math.max(0.2, Math.min(1, m.r2)),
      Math.max(0.2, Math.min(1, m.nse)),
      Math.max(0.2, Math.min(1, m.kge)),
      Math.max(0.2, Math.min(1, 1 - m.mape / 100)),
      Math.max(0.2, Math.min(1, m.adjR2)),
    ];
  };

  return (
    <div className="space-y-6">
      {/* Section 1: 9-Parameter Hyperparameter Console */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex items-center justify-between gap-4 pb-3 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-900">
            01. Live Hyperparameter &amp; Regularization Control Console (N = {analysis.nTotal.toLocaleString()})
          </h2>
          <button
            onClick={() =>
              onChangeHyper({
                trainSplitPct: 80,
                ridgeLambda: 1.5,
                l1Alpha: 0.4,
                gbdtTrees: 25,
                gbdtLearningRate: 0.16,
                gbdtMaxDepth: 2,
                rfTrees: 20,
                knnNeighbors: 7,
                classificationThreshold: 0.5,
              })
            }
            className="px-3 py-1 text-xs font-medium text-slate-700 bg-slate-100 rounded hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
          >
            Reset Optimal Defaults
          </button>
        </div>

        <div className="mt-4 grid grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-3">
          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Train Split</span>
              <span className="font-mono font-bold text-blue-700">{hyper.trainSplitPct}%</span>
            </div>
            <input
              type="range"
              min={60}
              max={90}
              step={5}
              value={hyper.trainSplitPct}
              onChange={(e) =>
                onChangeHyper({ ...hyper, trainSplitPct: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Ridge L2 (λ)</span>
              <span className="font-mono font-bold text-blue-700">{hyper.ridgeLambda}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={25}
              step={0.5}
              value={hyper.ridgeLambda}
              onChange={(e) =>
                onChangeHyper({ ...hyper, ridgeLambda: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Lasso L1 (α)</span>
              <span className="font-mono font-bold text-blue-700">{hyper.l1Alpha}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={3.0}
              step={0.1}
              value={hyper.l1Alpha}
              onChange={(e) =>
                onChangeHyper({ ...hyper, l1Alpha: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">GBDT Trees</span>
              <span className="font-mono font-bold text-blue-700">{hyper.gbdtTrees}</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              step={5}
              value={hyper.gbdtTrees}
              onChange={(e) =>
                onChangeHyper({ ...hyper, gbdtTrees: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Learn Rate η</span>
              <span className="font-mono font-bold text-blue-700">{hyper.gbdtLearningRate}</span>
            </div>
            <input
              type="range"
              min={0.04}
              max={0.4}
              step={0.02}
              value={hyper.gbdtLearningRate}
              onChange={(e) =>
                onChangeHyper({ ...hyper, gbdtLearningRate: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Tree Depth d</span>
              <span className="font-mono font-bold text-blue-700">{hyper.gbdtMaxDepth}</span>
            </div>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={hyper.gbdtMaxDepth}
              onChange={(e) =>
                onChangeHyper({ ...hyper, gbdtMaxDepth: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">RF Trees B</span>
              <span className="font-mono font-bold text-blue-700">{hyper.rfTrees}</span>
            </div>
            <input
              type="range"
              min={5}
              max={45}
              step={5}
              value={hyper.rfTrees}
              onChange={(e) =>
                onChangeHyper({ ...hyper, rfTrees: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">k-NN (k)</span>
              <span className="font-mono font-bold text-blue-700">{hyper.knnNeighbors}</span>
            </div>
            <input
              type="range"
              min={2}
              max={25}
              step={1}
              value={hyper.knnNeighbors}
              onChange={(e) =>
                onChangeHyper({ ...hyper, knnNeighbors: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[11px]">
              <span className="font-medium text-slate-700">Cutoff (τ)</span>
              <span className="font-mono font-bold text-blue-700">
                {hyper.classificationThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.15}
              max={0.85}
              step={0.05}
              value={hyper.classificationThreshold}
              onChange={(e) =>
                onChangeHyper({ ...hyper, classificationThreshold: Number(e.target.value) })
              }
              className="mt-1.5 w-full accent-blue-600"
            />
          </div>
        </div>
      </section>

      {/* Section 2: 5-Model Benchmark Table + Multi-Axis Radar Plot + Bias-Variance Learning Curve */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              02. 5-Model Quantitative Evaluation Benchmark (Test Set n = {analysis.nTest})
            </h2>
            <span className="text-xs text-slate-500">Click row to inspect model</span>
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-medium">
                  <th className="py-2 pr-2">Algorithm</th>
                  <th className="py-2 px-2 font-mono">Complexity</th>
                  <th className="py-2 px-2 text-right font-mono">RMSE</th>
                  <th className="py-2 px-2 text-right font-mono">MAE</th>
                  <th className="py-2 px-2 text-right font-mono">MAPE</th>
                  <th className="py-2 px-2 text-right font-mono">R²</th>
                  <th className="py-2 px-2 text-right font-mono">NSE</th>
                  <th className="py-2 pl-2 text-right font-mono">KGE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                {(['gbdt', 'ridge', 'elastic', 'rf', 'knn'] as const).map((mKey) => {
                  const m = analysis.models[mKey];
                  const isSelected = selectedModel === mKey;
                  return (
                    <tr
                      key={mKey}
                      onClick={() => onSelectModel(mKey)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-blue-50/80 font-semibold' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2.5 pr-2 font-sans font-medium text-slate-900">
                        {m.shortName} {isSelected ? '●' : ''}
                      </td>
                      <td className="py-2.5 px-2 text-slate-500 text-[11px]">{m.complexityNotation}</td>
                      <td className="py-2.5 px-2 text-right text-slate-900">{m.rmse.toFixed(1)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-700">{m.mae.toFixed(1)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-700">{m.mape.toFixed(2)}%</td>
                      <td className="py-2.5 px-2 text-right text-blue-700 font-bold">
                        {(m.r2 * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-2 text-right text-emerald-700 font-bold">
                        {m.nse.toFixed(3)}
                      </td>
                      <td className="py-2.5 pl-2 text-right text-slate-800">{m.kge.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Multi-Axis Radar / Spider Chart + Learning Curve */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200">
            <h2 className="text-base font-semibold text-slate-900">
              03. Skill Radar Polygon &amp; Sample Learning Curve
            </h2>
          </div>

          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
            <svg viewBox="0 0 210 190" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              {[0.35, 0.68, 1.0].map((rFrac) => {
                const pts = radarAxes
                  .map((ax) => {
                    const x = 105 + Math.cos(ax.angle) * 66 * rFrac;
                    const y = 98 + Math.sin(ax.angle) * 66 * rFrac;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');
                return (
                  <polygon
                    key={rFrac}
                    points={pts}
                    fill="none"
                    stroke="#cbd5e1"
                    strokeDasharray="2 2"
                  />
                );
              })}

              {radarAxes.map((ax) => {
                const x = 105 + Math.cos(ax.angle) * 66;
                const y = 98 + Math.sin(ax.angle) * 66;
                const lx = 105 + Math.cos(ax.angle) * 84;
                const ly = 98 + Math.sin(ax.angle) * 80;
                return (
                  <g key={ax.label}>
                    <line x1="105" y1="98" x2={x} y2={y} stroke="#cbd5e1" />
                    <text
                      x={lx}
                      y={ly + 3}
                      textAnchor="middle"
                      className="text-[8.5px] font-mono fill-slate-600 font-semibold"
                    >
                      {ax.label}
                    </text>
                  </g>
                );
              })}

              {(() => {
                const scores = getModelRadarScores(selectedModel);
                const pts = scores
                  .map((sc, i) => {
                    const x = 105 + Math.cos(radarAxes[i].angle) * 66 * sc;
                    const y = 98 + Math.sin(radarAxes[i].angle) * 66 * sc;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');
                return (
                  <polygon
                    points={pts}
                    fill="rgba(37, 99, 235, 0.22)"
                    stroke="#2563eb"
                    strokeWidth="2"
                  />
                );
              })()}
            </svg>

            <svg viewBox="0 0 210 190" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              <line x1="34" y1="16" x2="34" y2="162" stroke="#cbd5e1" />
              <line x1="34" y1="162" x2="195" y2="162" stroke="#cbd5e1" />

              {(() => {
                const lc = analysis.learningCurve;
                const allErr = lc.flatMap((p) => [p.trainRmse, p.valRmse]);
                const minE = Math.min(...allErr) * 0.85;
                const maxE = Math.max(...allErr) * 1.08;
                const scaleLX = (i: number) => 38 + (i / (lc.length - 1)) * 150;
                const scaleLY = (v: number) =>
                  158 - ((v - minE) / Math.max(1, maxE - minE)) * 135;

                const trainPath = lc
                  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleLX(i).toFixed(1)} ${scaleLY(p.trainRmse).toFixed(1)}`)
                  .join(' ');
                const valPath = lc
                  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleLX(i).toFixed(1)} ${scaleLY(p.valRmse).toFixed(1)}`)
                  .join(' ');

                return (
                  <>
                    <path d={valPath} fill="none" stroke="#dc2626" strokeWidth="2" />
                    <path d={trainPath} fill="none" stroke="#2563eb" strokeWidth="2" strokeDasharray="3 2" />
                    {lc.map((p, i) => (
                      <g key={p.sampleFractionPct}>
                        <circle cx={scaleLX(i)} cy={scaleLY(p.valRmse)} r="2.5" fill="#dc2626" />
                        <circle cx={scaleLX(i)} cy={scaleLY(p.trainRmse)} r="2.5" fill="#2563eb" />
                      </g>
                    ))}
                  </>
                );
              })()}

              <text x="115" y="180" textAnchor="middle" className="text-[8.5px] font-mono fill-slate-600">
                Learning Curve: Blue=Train, Red=Val RMSE
              </text>
            </svg>
          </div>
        </div>
      </section>

      {/* Section 3: Parity Plot & Residual Diagnostic */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              04. Predicted (ŷ) vs. Observed (y) Parity ({activeModel.shortName})
            </h2>
            <span className="text-xs font-mono text-slate-500">
              R² = {activeModel.r2.toFixed(4)} · NSE = {activeModel.nse.toFixed(4)}
            </span>
          </div>

          <div className="mt-3">
            <svg viewBox="0 0 420 235" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              <line x1="52" y1="16" x2="52" y2="205" stroke="#cbd5e1" />
              <line x1="52" y1="205" x2="395" y2="205" stroke="#cbd5e1" />

              <line
                x1="52"
                y1="205"
                x2="395"
                y2="16"
                stroke="#0f172a"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />

              {preds.map((p) => {
                const cx = 52 + ((p.actual - minVal) / Math.max(1, maxVal - minVal)) * 343;
                const cy = 205 - ((p.predicted - minVal) / Math.max(1, maxVal - minVal)) * 189;
                return (
                  <circle
                    key={p.id}
                    cx={Math.max(52, Math.min(395, cx))}
                    cy={Math.max(16, Math.min(205, cy))}
                    r="2.6"
                    fill={p.actualCritical ? '#dc2626' : '#2563eb'}
                    fillOpacity="0.58"
                  />
                );
              })}

              <text x="225" y="226" textAnchor="middle" className="text-[10px] font-mono fill-slate-600">
                Observed {problem.targetSymbol} ({problem.targetUnit})
              </text>
            </svg>
          </div>
        </div>

        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              05. Homoscedasticity Residual Diagnostic (eᵢ = yᵢ - ŷᵢ)
            </h2>
            <span className="text-xs font-mono text-slate-500">
              RMSE = ±{activeModel.rmse.toFixed(1)} {problem.targetUnit}
            </span>
          </div>

          <div className="mt-3">
            <svg viewBox="0 0 420 235" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              <line x1="52" y1="16" x2="52" y2="205" stroke="#cbd5e1" />
              <line x1="52" y1="205" x2="395" y2="205" stroke="#cbd5e1" />

              <line
                x1="52"
                y1="110.5"
                x2="395"
                y2="110.5"
                stroke="#dc2626"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />

              {preds.map((p) => {
                const cx = 52 + ((p.predicted - minVal) / Math.max(1, maxVal - minVal)) * 343;
                const cy = 110.5 - (p.residual / maxAbsRes) * 86;
                return (
                  <circle
                    key={p.id}
                    cx={Math.max(52, Math.min(395, cx))}
                    cy={Math.max(18, Math.min(202, cy))}
                    r="2.6"
                    fill="#0f172a"
                    fillOpacity="0.55"
                  />
                );
              })}

              <text x="225" y="226" textAnchor="middle" className="text-[10px] font-mono fill-slate-600">
                Predicted ŷ ({problem.targetUnit}) vs Residual Error (yᵢ - ŷᵢ)
              </text>
            </svg>
          </div>
        </div>
      </section>

      {/* Section 4: ROC / PR / Calibration Switcher + Confusion Matrix + 10-Feature Ridge & ElasticNet Table */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                06. ROC / PR / Calibration &amp; Confusion Matrix
              </h2>
              <p className="text-xs text-slate-500 mt-0.5 font-mono">
                ROC-AUC = {analysis.classificationEval.rocAuc.toFixed(3)} · PR-AUC ={' '}
                {analysis.classificationEval.prAuc.toFixed(3)}
              </p>
            </div>
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded">
              {(
                [
                  { id: 'roc', label: 'ROC' },
                  { id: 'pr', label: 'PR' },
                  { id: 'calibration', label: 'Calib' },
                ] as const
              ).map((t) => (
                <button
                  key={t.id}
                  onClick={() => setClassChartMode(t.id)}
                  className={`px-2 py-0.5 text-[11px] font-mono font-medium rounded ${
                    classChartMode === t.id ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-600'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
            <svg viewBox="0 0 200 200" className="w-full h-auto bg-slate-50 rounded border border-slate-200">
              <line x1="28" y1="18" x2="28" y2="172" stroke="#cbd5e1" />
              <line x1="28" y1="172" x2="182" y2="172" stroke="#cbd5e1" />

              {classChartMode === 'roc' && (
                <>
                  <line x1="28" y1="172" x2="182" y2="18" stroke="#94a3b8" strokeDasharray="3 3" />
                  <path
                    d={analysis.classificationEval.rocCurve
                      .map((pt, idx) => {
                        const x = 28 + pt.fpr * 154;
                        const y = 172 - pt.tpr * 154;
                        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(' ')}
                    fill="none"
                    stroke="#2563eb"
                    strokeWidth="2.2"
                  />
                  <text x="105" y="192" textAnchor="middle" className="text-[9px] font-mono fill-slate-600">
                    FPR vs Recall (TPR)
                  </text>
                </>
              )}

              {classChartMode === 'pr' && (
                <>
                  <path
                    d={[...analysis.classificationEval.rocCurve]
                      .sort((a, b) => a.tpr - b.tpr)
                      .map((pt, idx) => {
                        const x = 28 + pt.tpr * 154;
                        const y = 172 - pt.precision * 154;
                        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(' ')}
                    fill="none"
                    stroke="#059669"
                    strokeWidth="2.2"
                  />
                  <text x="105" y="192" textAnchor="middle" className="text-[9px] font-mono fill-slate-600">
                    Recall (TPR) vs Precision
                  </text>
                </>
              )}

              {classChartMode === 'calibration' && (
                <>
                  <line x1="28" y1="172" x2="182" y2="18" stroke="#94a3b8" strokeDasharray="3 3" />
                  <path
                    d={analysis.classificationEval.calibrationBins
                      .map((b, idx) => {
                        const x = 28 + b.meanPredictedProb * 154;
                        const y = 172 - b.empiricalFractionPos * 154;
                        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(' ')}
                    fill="none"
                    stroke="#d97706"
                    strokeWidth="2.2"
                  />
                  {analysis.classificationEval.calibrationBins.map((b, idx) => (
                    <circle
                      key={idx}
                      cx={28 + b.meanPredictedProb * 154}
                      cy={172 - b.empiricalFractionPos * 154}
                      r="3.2"
                      fill="#d97706"
                    />
                  ))}
                  <text x="105" y="192" textAnchor="middle" className="text-[9px] font-mono fill-slate-600">
                    Predicted Prob vs Observed Freq
                  </text>
                </>
              )}
            </svg>

            <div className="space-y-2 font-mono text-xs">
              <div className="text-slate-700 font-sans font-semibold">
                Confusion Matrix (τ = {analysis.classificationEval.threshold.toFixed(2)})
              </div>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded">
                  <div className="text-[10px] text-emerald-700">TP</div>
                  <div className="text-lg font-bold text-emerald-900">
                    {analysis.classificationEval.tp}
                  </div>
                </div>
                <div className="p-2.5 bg-amber-50 border border-amber-200 rounded">
                  <div className="text-[10px] text-amber-700">FP</div>
                  <div className="text-lg font-bold text-amber-900">
                    {analysis.classificationEval.fp}
                  </div>
                </div>
                <div className="p-2.5 bg-red-50 border border-red-200 rounded">
                  <div className="text-[10px] text-red-700">FN</div>
                  <div className="text-lg font-bold text-red-900">
                    {analysis.classificationEval.fn}
                  </div>
                </div>
                <div className="p-2.5 bg-blue-50 border border-blue-200 rounded">
                  <div className="text-[10px] text-blue-700">TN</div>
                  <div className="text-lg font-bold text-blue-900">
                    {analysis.classificationEval.tn}
                  </div>
                </div>
              </div>

              <div className="pt-1.5 grid grid-cols-2 gap-1 text-[11px] text-slate-700">
                <div>Precision: {(analysis.classificationEval.precision * 100).toFixed(1)}%</div>
                <div>Recall: {(analysis.classificationEval.recall * 100).toFixed(1)}%</div>
                <div>F1-Score: {analysis.classificationEval.f1Score.toFixed(3)}</div>
                <div>MCC: {analysis.classificationEval.mcc.toFixed(3)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* 10-Feature Ridge & ElasticNet Parametric Table */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
            <h2 className="text-base font-semibold text-slate-900">
              07. 10-Feature Ridge (L2) &amp; ElasticNet (L1/L2) Parametric Coefficients &amp; t-Tests
            </h2>
            <span className="text-xs font-mono text-slate-500">
              β₀ = {analysis.ridgeAnalytical.intercept.toFixed(1)}
            </span>
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-medium">
                  <th className="py-1.5 pr-2">Feature</th>
                  <th className="py-1.5 px-1.5 font-mono">Sym</th>
                  <th className="py-1.5 px-1.5 text-right font-mono">Ridge β̂ⱼ</th>
                  <th className="py-1.5 px-1.5 text-right font-mono">Elastic β̂ⱼ</th>
                  <th className="py-1.5 px-1.5 text-right font-mono">Std. β̂ⱼ (Z)</th>
                  <th className="py-1.5 px-1.5 text-right font-mono">SE(β̂ⱼ)</th>
                  <th className="py-1.5 px-1.5 text-right font-mono">t-Stat</th>
                  <th className="py-1.5 pl-1.5 text-right font-mono">p-Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono tabular-nums text-[11px]">
                {problem.features.map((f, j) => {
                  const rawB = analysis.ridgeAnalytical.rawBetas[j];
                  const elB = analysis.ridgeAnalytical.elasticBetas[j];
                  const stdB = analysis.ridgeAnalytical.standardizedBetas[j];
                  const se = analysis.ridgeAnalytical.stdErrors[j];
                  const tStat = analysis.ridgeAnalytical.tStats[j];
                  const pVal = analysis.ridgeAnalytical.pValues[j];
                  return (
                    <tr key={f.key} className="hover:bg-slate-50">
                      <td className="py-1.5 pr-2 font-sans font-medium text-slate-900">
                        {f.label}
                      </td>
                      <td className="py-1.5 px-1.5 text-blue-700 font-semibold">{f.symbol}</td>
                      <td className="py-1.5 px-1.5 text-right font-semibold text-slate-900">
                        {rawB >= 0 ? '+' : ''}
                        {rawB.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-1.5 text-right text-slate-700">
                        {elB >= 0 ? '+' : ''}
                        {elB.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-1.5 text-right text-slate-700">
                        {stdB >= 0 ? '+' : ''}
                        {stdB.toFixed(1)}
                      </td>
                      <td className="py-1.5 px-1.5 text-right text-slate-500">±{se.toFixed(2)}</td>
                      <td className="py-1.5 px-1.5 text-right font-semibold text-slate-900">
                        {tStat.toFixed(1)}
                      </td>
                      <td className="py-1.5 pl-1.5 text-right text-emerald-700 font-semibold">
                        {pVal < 0.001 ? '< 0.0001' : pVal.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
};
