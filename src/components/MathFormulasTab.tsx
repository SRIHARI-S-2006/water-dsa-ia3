import React, { useState } from 'react';
import { SIHProblemConfig } from '../data/sihDatasets';
import { EcosystemAnalysisResult, Hyperparameters } from '../engine/dataScienceEngine';

interface MathFormulasTabProps {
  problem: SIHProblemConfig;
  analysis: EcosystemAnalysisResult;
  hyper: Hyperparameters;
}

export const MathFormulasTab: React.FC<MathFormulasTabProps> = ({
  problem,
  analysis,
  hyper,
}) => {
  const [gLeft, setGLeft] = useState<number>(-420);
  const [hLeft, setHLeft] = useState<number>(85);
  const [gRight, setGRight] = useState<number>(380);
  const [hRight, setHRight] = useState<number>(92);
  const [gammaComplexity, setGammaComplexity] = useState<number>(15);

  const wLeftOpt = -gLeft / (hLeft + hyper.ridgeLambda);
  const wRightOpt = -gRight / (hRight + hyper.ridgeLambda);
  const scoreLeft = (gLeft * gLeft) / (hLeft + hyper.ridgeLambda);
  const scoreRight = (gRight * gRight) / (hRight + hyper.ridgeLambda);
  const scoreRoot =
    Math.pow(gLeft + gRight, 2) / (hLeft + hRight + hyper.ridgeLambda);
  const splitGain = 0.5 * (scoreLeft + scoreRight - scoreRoot) - gammaComplexity;

  return (
    <div className="space-y-6">
      {/* Algorithm 01: 10x10 L2 Ridge & L1 ElasticNet Normal Equations */}
      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-900">
            01. 10×10 Regularized Normal Equations (ZᵀZ + λI) &amp; Solved Vector β̂_std
          </h3>
          <span className="text-xs font-mono text-slate-500">
            n_train = {analysis.nTrain} · λ = {hyper.ridgeLambda} · α_L1 = {hyper.l1Alpha}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-4 p-4 bg-slate-900 text-slate-100 rounded-lg font-mono text-xs space-y-2 leading-relaxed">
            <div className="text-slate-400">Ridge (L2) Normal Equation:</div>
            <div className="text-blue-300 font-semibold">
              β̂_Ridge = (ZᵀZ + λI₁₀)⁻¹ Zᵀ(y - ȳ)
            </div>
            <div className="text-slate-400 pt-1">ElasticNet Soft-Thresholding:</div>
            <div className="text-emerald-300 font-semibold">
              β̂_j = S(β̂_OLS,j, α·λ₁) / (1 + λ₂)
            </div>
            <div className="text-slate-400 pt-1">Multicollinearity VIF:</div>
            <div>VIFⱼ = 1 / (1 - Rⱼ²) = (R₁₀×₁₀⁻¹)ⱼⱼ</div>
            <div className="text-slate-400 pt-1">Coupled SCS-CN Runoff Term:</div>
            <div>S_CN = 25400/CN - 254, Q_r = (R - 0.2S)²/(R + 0.8S)</div>
          </div>

          <div className="lg:col-span-8 overflow-x-auto border border-slate-200 rounded-lg">
            <table className="w-full text-right border-collapse text-[11px] font-mono tabular-nums">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                  <th className="p-1.5 text-left">Sym</th>
                  {problem.features.map((f) => (
                    <th key={f.key} className="p-1.5">
                      {f.symbol}
                    </th>
                  ))}
                  <th className="p-1.5 bg-blue-50 text-blue-900 border-l border-slate-200">
                    Zᵀy
                  </th>
                  <th className="p-1.5 bg-emerald-50 text-emerald-900 border-l border-slate-200">
                    β̂_std
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {analysis.ridgeAnalytical.xtxMatrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-1.5 text-left font-semibold text-slate-900 bg-slate-50">
                      {problem.features[rIdx].symbol}
                    </td>
                    {row.map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`p-1.5 ${
                          rIdx === cIdx ? 'font-bold text-slate-900 bg-slate-100/70' : 'text-slate-600'
                        }`}
                      >
                        {val.toFixed(0)}
                      </td>
                    ))}
                    <td className="p-1.5 bg-blue-50/50 text-blue-900 font-medium border-l border-slate-200">
                      {(analysis.ridgeAnalytical.xtyVector[rIdx] / 1000).toFixed(1)}k
                    </td>
                    <td className="p-1.5 bg-emerald-50/50 text-emerald-900 font-bold border-l border-slate-200">
                      {analysis.ridgeAnalytical.standardizedBetas[rIdx] >= 0 ? '+' : ''}
                      {analysis.ridgeAnalytical.standardizedBetas[rIdx].toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Algorithm 02: Second-Order GBDT / XGBoost & Node Split Calculator */}
      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-900">
            02. Second-Order Gradient Boosted Trees (GBDT / XGBoost) &amp; Node Split Calculator
          </h3>
          <span className="text-xs font-mono text-slate-500">
            M = {hyper.gbdtTrees} · η = {hyper.gbdtLearningRate} · d = {hyper.gbdtMaxDepth}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 p-4 bg-slate-900 text-slate-100 rounded-lg font-mono text-xs space-y-2 leading-relaxed">
            <div className="text-slate-400">Second-Order Taylor Objective:</div>
            <div>
              L⁽ᵗ⁾ ≈ ∑ᵢ [ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢ fₜ(xᵢ) + ½ hᵢ fₜ²(xᵢ) ] + γT + ½λ ∑ⱼ wⱼ²
            </div>
            <div className="text-blue-300 font-semibold pt-1">
              Optimal Leaf Weight: wⱼ* = - Gⱼ / (Hⱼ + λ)
            </div>
            <div className="text-emerald-300 font-semibold pt-1">
              Split Gain = ½ [ G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ) ] - γ
            </div>
          </div>

          <div className="lg:col-span-6 p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3">
            <div className="grid grid-cols-5 gap-2 font-mono text-xs">
              <div>
                <label className="block text-[10px] text-slate-500">G_L</label>
                <input
                  type="number"
                  value={gLeft}
                  onChange={(e) => setGLeft(Number(e.target.value))}
                  className="mt-1 w-full px-2 py-1 bg-white border border-slate-300 rounded"
                />
              </div>
              <div>
                <label className="block text-[10px] text-slate-500">H_L</label>
                <input
                  type="number"
                  value={hLeft}
                  onChange={(e) => setHLeft(Math.max(1, Number(e.target.value)))}
                  className="mt-1 w-full px-2 py-1 bg-white border border-slate-300 rounded"
                />
              </div>
              <div>
                <label className="block text-[10px] text-slate-500">G_R</label>
                <input
                  type="number"
                  value={gRight}
                  onChange={(e) => setGRight(Number(e.target.value))}
                  className="mt-1 w-full px-2 py-1 bg-white border border-slate-300 rounded"
                />
              </div>
              <div>
                <label className="block text-[10px] text-slate-500">H_R</label>
                <input
                  type="number"
                  value={hRight}
                  onChange={(e) => setHRight(Math.max(1, Number(e.target.value)))}
                  className="mt-1 w-full px-2 py-1 bg-white border border-slate-300 rounded"
                />
              </div>
              <div>
                <label className="block text-[10px] text-slate-500">γ</label>
                <input
                  type="number"
                  value={gammaComplexity}
                  onChange={(e) => setGammaComplexity(Number(e.target.value))}
                  className="mt-1 w-full px-2 py-1 bg-white border border-slate-300 rounded"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 font-mono text-xs">
              <div className="p-2 bg-white border border-slate-200 rounded">
                <div className="text-[10px] text-slate-500">Left Weight w_L*</div>
                <div className="text-sm font-bold text-slate-900 mt-0.5">
                  {wLeftOpt >= 0 ? '+' : ''}
                  {wLeftOpt.toFixed(3)}
                </div>
              </div>
              <div className="p-2 bg-white border border-slate-200 rounded">
                <div className="text-[10px] text-slate-500">Right Weight w_R*</div>
                <div className="text-sm font-bold text-slate-900 mt-0.5">
                  {wRightOpt >= 0 ? '+' : ''}
                  {wRightOpt.toFixed(3)}
                </div>
              </div>
              <div className="p-2 bg-white border border-slate-200 rounded">
                <div className="text-[10px] text-slate-500">Exact Split Gain</div>
                <div
                  className={`text-sm font-bold mt-0.5 ${
                    splitGain > 0 ? 'text-emerald-700' : 'text-red-600'
                  }`}
                >
                  {splitGain.toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Algorithm 03 & 04: Gumbel Return Period Table (T = 2 to 1000 yrs) & 10D PCA Spectral Table */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="pb-2.5 border-b border-slate-200 flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-900">
              03. Gumbel EV1 Return Period Quantiles
            </h3>
            <span className="text-xs font-mono text-slate-500">
              μ={analysis.gumbel.muLocation.toFixed(0)} · α={analysis.gumbel.alphaScale.toFixed(0)}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5">Return Period T</th>
                  <th className="py-1.5 text-right">P = 1/T</th>
                  <th className="py-1.5 text-right">Reduced y_T*</th>
                  <th className="py-1.5 text-right">Peak Q_T (m³/s)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {analysis.gumbel.returnPeriods.map((rp) => (
                  <tr key={rp.returnPeriodYears} className="hover:bg-slate-50">
                    <td className="py-1.5 font-bold text-slate-900">
                      {rp.returnPeriodYears}-Year Flood
                    </td>
                    <td className="py-1.5 text-right text-slate-600">
                      {rp.exceedanceProbPct.toFixed(2)}%
                    </td>
                    <td className="py-1.5 text-right text-slate-600">
                      {rp.reducedVariateYt.toFixed(3)}
                    </td>
                    <td
                      className={`py-1.5 text-right font-bold ${
                        rp.exceedsCritical ? 'text-red-700' : 'text-blue-700'
                      }`}
                    >
                      {rp.expectedPeakValue.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="pb-2.5 border-b border-slate-200 flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-900">
              04. 10D Principal Component Orthogonal Eigenvectors (R v_k = λ_k v_k)
            </h3>
            <span className="text-xs font-mono text-slate-500">Jacobi Spectral Solver</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-right border-collapse text-[11px] font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5 text-left">PC</th>
                  <th className="py-1.5">λ_k</th>
                  <th className="py-1.5">EVR%</th>
                  {problem.features.map((f) => (
                    <th key={f.key} className="py-1.5">
                      {f.symbol}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {analysis.pca.eigenvalues.slice(0, 8).map((ev, k) => (
                  <tr key={k}>
                    <td className="py-1.5 text-left font-bold text-slate-900">PC{k + 1}</td>
                    <td className="py-1.5 text-blue-700 font-semibold">{ev.toFixed(2)}</td>
                    <td className="py-1.5 text-slate-700">
                      {analysis.pca.explainedVariancePct[k].toFixed(1)}%
                    </td>
                    {analysis.pca.loadings[k].map((loadVal, fIdx) => (
                      <td key={fIdx} className="py-1.5 text-slate-600">
                        {loadVal >= 0 ? '+' : ''}
                        {loadVal.toFixed(2)}
                      </td>
                    ))}
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
