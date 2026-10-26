import React, { useMemo, useRef, useState } from 'react';
import { DataRecord, FEATURE_KEYS, FeatureKey, SIHProblemConfig } from '../data/sihDatasets';
import { EcosystemAnalysisResult } from '../engine/dataScienceEngine';
import { Download, Plus, RotateCcw, Search, Sparkles, Upload } from 'lucide-react';

interface DatasetExplorerTabProps {
  problem: SIHProblemConfig;
  records: DataRecord[];
  analysis: EcosystemAnalysisResult;
  onAddRecord: (newRec: DataRecord) => void;
  onAddBatchRecords: (batch: DataRecord[]) => void;
  onResetDataset: () => void;
  onExportCsv: () => void;
}

export const DatasetExplorerTab: React.FC<DatasetExplorerTabProps> = ({
  problem,
  records,
  analysis,
  onAddRecord,
  onAddBatchRecords,
  onResetDataset,
  onExportCsv,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [regimeFilter, setRegimeFilter] = useState<'ALL' | 'Severe' | 'Advisory' | 'Normal' | 'OUTLIERS'>('ALL');
  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState<number>(0); // 0..9 = x1..x10, 10 = target
  const [sortCol, setSortCol] = useState<'id' | FeatureKey | 'target'>('id');
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [showAddForm, setShowAddForm] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [newStation, setNewStation] = useState<string>(problem.stations[0]);
  const [customInputs, setCustomInputs] = useState<Record<FeatureKey, number>>({
    x1: problem.features[0].defaultVal,
    x2: problem.features[1].defaultVal,
    x3: problem.features[2].defaultVal,
    x4: problem.features[3].defaultVal,
    x5: problem.features[4].defaultVal,
    x6: problem.features[5].defaultVal,
    x7: problem.features[6].defaultVal,
    x8: problem.features[7].defaultVal,
    x9: problem.features[8].defaultVal,
    x10: problem.features[9].defaultVal,
  });

  const pageSize = 30;

  const activeStat =
    selectedFeatureIdx < 10
      ? analysis.featureStats[selectedFeatureIdx]
      : analysis.targetStats;

  const activeValues = useMemo(() => {
    if (selectedFeatureIdx < 10) {
      const k = problem.features[selectedFeatureIdx].key;
      return records.map((r) => r[k]);
    }
    return records.map((r) => r.target);
  }, [records, selectedFeatureIdx, problem]);

  const histogramBins = useMemo(() => {
    const numBins = 20;
    const minV = activeStat.min;
    const maxV = Math.max(minV + 1e-5, activeStat.max);
    const binWidth = (maxV - minV) / numBins;
    const counts = new Array(numBins).fill(0);
    for (const v of activeValues) {
      const idx = Math.min(numBins - 1, Math.max(0, Math.floor((v - minV) / binWidth)));
      counts[idx]++;
    }
    const maxCount = Math.max(1, ...counts);
    let cumCount = 0;
    return counts.map((count, i) => {
      cumCount += count;
      const binMid = minV + (i + 0.5) * binWidth;
      const z = (binMid - activeStat.mean) / Math.max(1e-6, activeStat.std);
      const gaussPdf = Math.exp(-0.5 * z * z);
      return {
        binStart: minV + i * binWidth,
        binEnd: minV + (i + 1) * binWidth,
        binMid,
        count,
        heightPct: (count / maxCount) * 100,
        ecdfPct: (cumCount / Math.max(1, activeValues.length)) * 100,
        gaussPct: gaussPdf * 92,
      };
    });
  }, [activeValues, activeStat]);

  const filteredRecords = useMemo(() => {
    const tMean = analysis.targetStats.mean;
    const tStd = analysis.targetStats.std;

    return records
      .filter((r) => {
        if (searchQuery.trim() !== '') {
          const q = searchQuery.toLowerCase();
          const match =
            r.id.toLowerCase().includes(q) ||
            r.station.toLowerCase().includes(q) ||
            r.seasonOrZone.toLowerCase().includes(q);
          if (!match) return false;
        }
        if (regimeFilter === 'Severe') return r.riskClass === 'Severe';
        if (regimeFilter === 'Advisory') return r.riskClass === 'Advisory';
        if (regimeFilter === 'Normal') return r.riskClass === 'Normal';
        if (regimeFilter === 'OUTLIERS') {
          const z = Math.abs((r.target - tMean) / Math.max(1e-6, tStd));
          return z >= 1.96;
        }
        return true;
      })
      .sort((a, b) => {
        const vA = a[sortCol];
        const vB = b[sortCol];
        if (vA < vB) return sortAsc ? -1 : 1;
        if (vA > vB) return sortAsc ? 1 : -1;
        return 0;
      });
  }, [records, searchQuery, regimeFilter, sortCol, sortAsc, analysis.targetStats]);

  const totalPages = Math.max(1, Math.ceil(filteredRecords.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageSlice = filteredRecords.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleSort = (col: typeof sortCol) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(false);
    }
  };

  const handleCreateCustomRecord = (e: React.FormEvent) => {
    e.preventDefault();
    const sim = analysis.predictSingle(customInputs);
    const target = Number(sim.ensembleMean.toFixed(0));
    const riskClass: 'Normal' | 'Advisory' | 'Severe' =
      target >= problem.criticalThreshold
        ? 'Severe'
        : target >= problem.advisoryThreshold
          ? 'Advisory'
          : 'Normal';

    const customRec: DataRecord = {
      id: `CUSTOM-${String(records.length + 1).padStart(4, '0')}`,
      timestamp: new Date().toISOString().slice(0, 16).replace('T', ' '),
      station: newStation,
      seasonOrZone: 'Live Telemetry Input',
      ...customInputs,
      waterLevelM: sim.estimatedWaterLevelM,
      target,
      riskClass,
      isCritical: riskClass === 'Severe' ? 1 : 0,
    };
    onAddRecord(customRec);
    setShowAddForm(false);
  };

  // Generate +500 synthetic bootstrap-perturbed records to expand dataset on demand
  const handleSynthesizeBatch = () => {
    const batch: DataRecord[] = [];
    const baseLen = records.length;
    for (let i = 0; i < 500; i++) {
      const ref = records[(i * 17 + 11) % records.length];
      const jitter = ((i % 13) - 6) * 0.022;
      const clampFeat = (val: number, fIdx: number) => {
        const sp = problem.features[fIdx];
        return Number(Math.max(sp.min, Math.min(sp.max, val * (1 + jitter))).toFixed(2));
      };
      const x1 = clampFeat(ref.x1, 0);
      const x2 = clampFeat(ref.x2, 1);
      const x3 = clampFeat(ref.x3, 2);
      const x4 = clampFeat(ref.x4, 3);
      const x5 = clampFeat(ref.x5, 4);
      const x6 = clampFeat(ref.x6, 5);
      const x7 = clampFeat(ref.x7, 6);
      const x8 = clampFeat(ref.x8, 7);
      const x9 = clampFeat(ref.x9, 8);
      const x10 = clampFeat(ref.x10, 9);
      const target = Number((ref.target * (1 + jitter * 0.85)).toFixed(0));
      const waterLevelM = Number((82.5 + Math.pow(Math.max(100, target) / 180, 0.68)).toFixed(2));
      const riskClass: 'Normal' | 'Advisory' | 'Severe' =
        target >= problem.criticalThreshold
          ? 'Severe'
          : target >= problem.advisoryThreshold
            ? 'Advisory'
            : 'Normal';

      batch.push({
        id: `SMOTE-${String(baseLen + i + 1).padStart(4, '0')}`,
        timestamp: ref.timestamp,
        station: ref.station,
        seasonOrZone: `${ref.seasonOrZone} (SMOTE)`,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        waterLevelM,
        target,
        riskClass,
        isCritical: riskClass === 'Severe' ? 1 : 0,
      });
    }
    onAddBatchRecords(batch);
  };

  const handleCsvUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = String(evt.target?.result || '');
      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      if (lines.length < 2) return;
      const imported: DataRecord[] = [];
      for (let i = 1; i < Math.min(lines.length, 1500); i++) {
        const cols = lines[i].split(',').map((c) => c.replace(/^"|"$/g, '').trim());
        if (cols.length < 11) continue;
        const offset = cols.length >= 15 ? 4 : 0;
        const x1 = Number(cols[offset]) || problem.features[0].defaultVal;
        const x2 = Number(cols[offset + 1]) || problem.features[1].defaultVal;
        const x3 = Number(cols[offset + 2]) || problem.features[2].defaultVal;
        const x4 = Number(cols[offset + 3]) || problem.features[3].defaultVal;
        const x5 = Number(cols[offset + 4]) || problem.features[4].defaultVal;
        const x6 = Number(cols[offset + 5]) || problem.features[5].defaultVal;
        const x7 = Number(cols[offset + 6]) || problem.features[6].defaultVal;
        const x8 = Number(cols[offset + 7]) || problem.features[7].defaultVal;
        const x9 = Number(cols[offset + 8]) || problem.features[8].defaultVal;
        const x10 = Number(cols[offset + 9]) || problem.features[9].defaultVal;
        const target = Number(cols[offset + 10]) || analysis.targetStats.mean;
        const waterLevelM = Number((82.5 + Math.pow(Math.max(100, target) / 180, 0.68)).toFixed(2));
        const riskClass: 'Normal' | 'Advisory' | 'Severe' =
          target >= problem.criticalThreshold
            ? 'Severe'
            : target >= problem.advisoryThreshold
              ? 'Advisory'
              : 'Normal';
        imported.push({
          id: cols[0] && isNaN(Number(cols[0])) ? cols[0] : `CSV-${String(records.length + i).padStart(4, '0')}`,
          timestamp: cols.length >= 15 ? cols[1] : new Date().toISOString().slice(0, 16).replace('T', ' '),
          station: cols.length >= 15 ? cols[2] : problem.stations[i % problem.stations.length],
          seasonOrZone: cols.length >= 15 ? cols[3] : 'CSV Imported Cohort',
          x1,
          x2,
          x3,
          x4,
          x5,
          x6,
          x7,
          x8,
          x9,
          x10,
          waterLevelM,
          target,
          riskClass,
          isCritical: riskClass === 'Severe' ? 1 : 0,
        });
      }
      if (imported.length > 0) {
        onAddBatchRecords(imported);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Section 1: 10-Feature Statistical Moments & VIF Matrix */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-900">
            01. 10-Feature Parametric &amp; Non-Parametric Distribution Matrix (N = {records.length.toLocaleString()})
          </h2>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              onClick={handleSynthesizeBatch}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors whitespace-nowrap shrink-0"
            >
              <Sparkles className="w-3.5 h-3.5" />
              +500 SMOTE Records
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleCsvUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
            >
              <Upload className="w-3.5 h-3.5" />
              Import CSV
            </button>
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0"
            >
              <Plus className="w-3.5 h-3.5" />
              {showAddForm ? 'Close Form' : 'Add Telemetry Row'}
            </button>
            <button
              onClick={onResetDataset}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset (2,500)
            </button>
          </div>
        </div>

        {showAddForm && (
          <form
            onSubmit={handleCreateCustomRecord}
            className="my-4 p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3"
          >
            <div className="flex items-center justify-between text-xs font-semibold text-slate-900">
              <span>Inject Custom 10-Feature Sensor Observation</span>
              <span className="font-mono text-slate-500">
                ID: CUSTOM-{String(records.length + 1).padStart(4, '0')}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5">
              <div>
                <label className="block text-[10px] font-medium text-slate-600 mb-1">
                  River Basin Station
                </label>
                <select
                  value={newStation}
                  onChange={(e) => setNewStation(e.target.value)}
                  className="w-full px-2 py-1.5 text-xs bg-white border border-slate-300 rounded font-medium"
                >
                  {problem.stations.map((st) => (
                    <option key={st} value={st}>
                      {st}
                    </option>
                  ))}
                </select>
              </div>
              {problem.features.map((spec) => (
                <div key={spec.key}>
                  <label className="block text-[10px] font-medium text-slate-600 mb-1 truncate">
                    {spec.symbol} · {spec.label} ({spec.unit})
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={customInputs[spec.key]}
                    onChange={(e) =>
                      setCustomInputs({ ...customInputs, [spec.key]: Number(e.target.value) })
                    }
                    className="w-full px-2 py-1.5 text-xs bg-white border border-slate-300 rounded font-mono tabular-nums"
                  />
                </div>
              ))}
            </div>

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-3 py-1 text-xs font-medium text-slate-600 hover:text-slate-900"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3.5 py-1 text-xs font-medium text-white bg-slate-900 rounded hover:bg-slate-800"
              >
                Append Row &amp; Retrain
              </button>
            </div>
          </form>
        )}

        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 font-medium">
                <th className="py-2 pr-3">Feature / Variable</th>
                <th className="py-2 px-2 font-mono">Symbol</th>
                <th className="py-2 px-2 text-right font-mono">Mean (μ)</th>
                <th className="py-2 px-2 text-right font-mono">Std (σ)</th>
                <th className="py-2 px-2 text-right font-mono">Min</th>
                <th className="py-2 px-2 text-right font-mono">Q₁ (25%)</th>
                <th className="py-2 px-2 text-right font-mono">Median</th>
                <th className="py-2 px-2 text-right font-mono">Q₃ (75%)</th>
                <th className="py-2 px-2 text-right font-mono">Max</th>
                <th className="py-2 px-2 text-right font-mono">Skew (γ₁)</th>
                <th className="py-2 px-2 text-right font-mono">Kurt (κ)</th>
                <th className="py-2 px-2 text-right font-mono">VIF</th>
                <th className="py-2 pl-2 text-right font-mono">r (Q_peak)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
              {[...analysis.featureStats, analysis.targetStats].map((st, idx) => {
                const isTarget = idx === 10;
                const isSelected = selectedFeatureIdx === idx;
                return (
                  <tr
                    key={st.key}
                    onClick={() => setSelectedFeatureIdx(idx)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-blue-50/70'
                        : isTarget
                          ? 'bg-slate-50 font-semibold'
                          : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-2 pr-3 font-sans font-medium text-slate-900">
                      {st.label} {isTarget ? '(Target y)' : ''}
                    </td>
                    <td className="py-2 px-2 text-blue-700 font-semibold">
                      {st.symbol} ({st.unit})
                    </td>
                    <td className="py-2 px-2 text-right text-slate-900">{st.mean.toFixed(2)}</td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.std.toFixed(2)}</td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.min.toFixed(1)}</td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.q25.toFixed(1)}</td>
                    <td className="py-2 px-2 text-right font-semibold text-slate-900">
                      {st.median.toFixed(1)}
                    </td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.q75.toFixed(1)}</td>
                    <td className="py-2 px-2 text-right text-slate-600">{st.max.toFixed(1)}</td>
                    <td className="py-2 px-2 text-right text-slate-700">
                      {st.skewness >= 0 ? '+' : ''}
                      {st.skewness.toFixed(3)}
                    </td>
                    <td className="py-2 px-2 text-right text-slate-700">
                      {st.kurtosis >= 0 ? '+' : ''}
                      {st.kurtosis.toFixed(3)}
                    </td>
                    <td className="py-2 px-2 text-right text-slate-700">
                      {isTarget ? '—' : st.vif.toFixed(2)}
                    </td>
                    <td className="py-2 pl-2 text-right font-semibold text-blue-700">
                      {st.pearsonWithTarget.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 2: 20-Bin Empirical Density Histogram, Gaussian PDF Fit & ECDF Curve */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-slate-900">
              02. Density Histogram, Gaussian Fit &amp; ECDF: {activeStat.label} ({activeStat.symbol})
            </h2>
            <span className="text-xs font-mono text-slate-500">
              IQR={activeStat.iqr.toFixed(1)} · Outliers={activeStat.outlierCount}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-lg">
            {[...problem.features.map((f, i) => ({ label: f.symbol, idx: i })), { label: problem.targetSymbol, idx: 10 }].map(
              (item) => (
                <button
                  key={item.idx}
                  onClick={() => setSelectedFeatureIdx(item.idx)}
                  className={`px-2 py-1 text-xs font-mono font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    selectedFeatureIdx === item.idx
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {item.label}
                </button>
              )
            )}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-8">
            <svg viewBox="0 0 600 195" className="w-full h-auto bg-slate-50 border border-slate-200 rounded">
              <line x1="42" y1="16" x2="42" y2="168" stroke="#cbd5e1" />
              <line x1="42" y1="168" x2="572" y2="168" stroke="#cbd5e1" />

              {histogramBins.map((b, i) => {
                const barW = 520 / histogramBins.length - 3;
                const x = 46 + i * (520 / histogramBins.length);
                const h = (b.heightPct / 100) * 140;
                const y = 168 - h;
                return (
                  <g key={i}>
                    <rect
                      x={x}
                      y={y}
                      width={barW}
                      height={Math.max(2, h)}
                      fill="#2563eb"
                      fillOpacity="0.8"
                      rx="1.5"
                    >
                      <title>
                        Bin [{b.binStart.toFixed(1)}, {b.binEnd.toFixed(1)}]: n = {b.count} | ECDF ={' '}
                        {b.ecdfPct.toFixed(1)}%
                      </title>
                    </rect>
                    {i % 2 === 0 && (
                      <text
                        x={x + barW / 2}
                        y="184"
                        textAnchor="middle"
                        className="text-[9px] font-mono fill-slate-500"
                      >
                        {b.binStart.toFixed(0)}
                      </text>
                    )}
                  </g>
                );
              })}

              <path
                d={histogramBins
                  .map((b, i) => {
                    const x = 46 + (i + 0.5) * (520 / histogramBins.length);
                    const y = 168 - (b.gaussPct / 100) * 140;
                    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                  })
                  .join(' ')}
                fill="none"
                stroke="#dc2626"
                strokeWidth="2"
              />

              <path
                d={histogramBins
                  .map((b, i) => {
                    const x = 46 + (i + 0.5) * (520 / histogramBins.length);
                    const y = 168 - (b.ecdfPct / 100) * 140;
                    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                  })
                  .join(' ')}
                fill="none"
                stroke="#0f172a"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
            </svg>
          </div>

          <div className="lg:col-span-4 space-y-2.5 font-mono text-xs bg-slate-50 p-4 rounded border border-slate-200">
            <div className="p-2.5 bg-slate-900 text-slate-100 rounded text-[11px] space-y-1">
              <div>f(x) = (1 / (σ√2π)) · exp(-½((x - μ)/σ)²)</div>
              <div className="text-blue-300">
                zᵢ = (xᵢ - {activeStat.mean.toFixed(2)}) / {activeStat.std.toFixed(2)}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 pt-1">
              <div>
                <span className="text-slate-500">CV (σ/μ):</span>
                <div className="font-bold text-slate-900">{activeStat.cvPct.toFixed(2)}%</div>
              </div>
              <div>
                <span className="text-slate-500">Variance (s²):</span>
                <div className="font-bold text-slate-900">{activeStat.variance.toFixed(1)}</div>
              </div>
              <div>
                <span className="text-slate-500">10th–90th Span:</span>
                <div className="font-bold text-slate-900">
                  {activeStat.q10.toFixed(1)} – {activeStat.q90.toFixed(1)}
                </div>
              </div>
              <div>
                <span className="text-slate-500">Pearson r:</span>
                <div className="font-bold text-blue-700">
                  {activeStat.pearsonWithTarget.toFixed(4)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: High-Density 10-Feature Telemetry Data Grid (2,500+ Rows) */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-900">
            03. Ground-Truth CWC River Basin Dataset ({filteredRecords.length.toLocaleString()} matching records)
          </h2>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setPage(1);
                }}
                placeholder="Search ID, basin, monsoon..."
                className="pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:border-blue-600 w-48"
              />
            </div>

            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {(
                [
                  { id: 'ALL', label: 'All' },
                  { id: 'Severe', label: 'Severe' },
                  { id: 'Advisory', label: 'Advisory' },
                  { id: 'Normal', label: 'Nominal' },
                  { id: 'OUTLIERS', label: '|z| ≥ 1.96' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => {
                    setRegimeFilter(tab.id);
                    setPage(1);
                  }}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    regimeFilter === tab.id
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <button
              onClick={onExportCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
        </div>

        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 font-medium select-none">
                <th
                  onClick={() => handleSort('id')}
                  className="py-2 pr-2 cursor-pointer hover:text-slate-900 whitespace-nowrap"
                >
                  ID {sortCol === 'id' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th className="py-2 px-1.5 whitespace-nowrap">River Basin Station</th>
                {problem.features.map((f) => (
                  <th
                    key={f.key}
                    onClick={() => handleSort(f.key)}
                    className="py-2 px-1.5 text-right font-mono cursor-pointer hover:text-slate-900 whitespace-nowrap"
                    title={`${f.label} (${f.unit})`}
                  >
                    {f.symbol} {sortCol === f.key ? (sortAsc ? '↑' : '↓') : ''}
                  </th>
                ))}
                <th className="py-2 px-1.5 text-right font-mono whitespace-nowrap">W_lvl (m)</th>
                <th
                  onClick={() => handleSort('target')}
                  className="py-2 px-2 text-right font-mono font-semibold text-slate-900 cursor-pointer whitespace-nowrap"
                >
                  Q_peak {sortCol === 'target' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th className="py-2 pl-2 text-right whitespace-nowrap">Alert Class</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono tabular-nums text-[11px]">
              {pageSlice.map((rec) => {
                const stateColor =
                  rec.riskClass === 'Severe'
                    ? 'text-red-700 font-semibold'
                    : rec.riskClass === 'Advisory'
                      ? 'text-amber-700 font-medium'
                      : 'text-emerald-700';

                return (
                  <tr key={rec.id} className="hover:bg-slate-50">
                    <td className="py-1.5 pr-2 font-semibold text-slate-900 whitespace-nowrap">
                      {rec.id}
                    </td>
                    <td className="py-1.5 px-1.5 font-sans text-slate-800 whitespace-nowrap">
                      {rec.station}
                    </td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x1}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x2}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x3}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x4}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x5}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x6}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x7}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x8}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x9}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-700">{rec.x10}</td>
                    <td className="py-1.5 px-1.5 text-right text-slate-600">{rec.waterLevelM}</td>
                    <td className="py-1.5 px-2 text-right font-bold text-slate-900">{rec.target}</td>
                    <td className={`py-1.5 pl-2 text-right font-sans whitespace-nowrap ${stateColor}`}>
                      {rec.riskClass}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-3 pt-2.5 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
          <div className="font-mono tabular-nums">
            Showing {(currentPage - 1) * pageSize + 1}–
            {Math.min(filteredRecords.length, currentPage * pageSize)} of{' '}
            {filteredRecords.length.toLocaleString()} records
          </div>
          <div className="flex items-center gap-2">
            <button
              disabled={currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1 border border-slate-200 rounded bg-white hover:bg-slate-50 disabled:opacity-40 font-medium"
            >
              Previous
            </button>
            <span className="font-mono tabular-nums px-2">
              Page {currentPage} / {totalPages}
            </span>
            <button
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1 border border-slate-200 rounded bg-white hover:bg-slate-50 disabled:opacity-40 font-medium"
            >
              Next
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
