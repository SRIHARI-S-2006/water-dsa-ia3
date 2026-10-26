/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useMemo, useState } from 'react';
import { SIH_PROBLEM, DataRecord } from './data/sihDatasets';
import { runEcosystemAnalysis, Hyperparameters, ModelId } from './engine/dataScienceEngine';
import { OverviewTab } from './components/OverviewTab';
import { DatasetExplorerTab } from './components/DatasetExplorerTab';
import { ModelStudioTab } from './components/ModelStudioTab';
import { MathFormulasTab } from './components/MathFormulasTab';
import { SimulatorTab } from './components/SimulatorTab';
import {
  Cpu,
  Database,
  Download,
  LayoutDashboard,
  Sigma,
  Sliders,
  Waves,
} from 'lucide-react';

type TabId = 'overview' | 'dataset' | 'models' | 'math' | 'simulator';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [selectedModel, setSelectedModel] = useState<ModelId>('gbdt');

  // Dataset state initialized with the 2,500 multi-basin CWC flood telemetry records
  const [records, setRecords] = useState<DataRecord[]>(() => SIH_PROBLEM.records);

  // Live ML & Statistical Hyperparameters
  const [hyper, setHyper] = useState<Hyperparameters>({
    trainSplitPct: 80,
    ridgeLambda: 1.5,
    l1Alpha: 0.4,
    gbdtTrees: 35,
    gbdtLearningRate: 0.12,
    gbdtMaxDepth: 3,
    rfTrees: 16,
    knnNeighbors: 7,
    classificationThreshold: 0.5,
  });

  // Execute the live Data Science & Mathematical Pipeline whenever records or hyperparameters change
  const analysis = useMemo(() => {
    return runEcosystemAnalysis(SIH_PROBLEM, records, hyper);
  }, [records, hyper]);

  const handleAddRecord = (newRec: DataRecord) => {
    setRecords((prev) => [newRec, ...prev]);
  };

  const handleAddBatchRecords = (batch: DataRecord[]) => {
    setRecords((prev) => [...batch, ...prev]);
  };

  const handleResetDataset = () => {
    setRecords(SIH_PROBLEM.records);
  };

  const handleExportCsv = () => {
    const headers = [
      'ID',
      'Timestamp',
      'Basin_Station',
      'Hydro_Zone',
      ...SIH_PROBLEM.features.map((f) => `${f.key}_${f.code}`),
      'WaterLevel_m',
      `Target_${SIH_PROBLEM.targetSymbol}`,
      'Risk_Class',
      'Is_Critical',
    ];
    const rowsCsv = records.map((r) =>
      [
        r.id,
        r.timestamp,
        `"${r.station}"`,
        `"${r.seasonOrZone}"`,
        r.x1,
        r.x2,
        r.x3,
        r.x4,
        r.x5,
        r.x6,
        r.x7,
        r.x8,
        r.x9,
        r.x10,
        r.waterLevelM,
        r.target,
        r.riskClass,
        r.isCritical,
      ].join(',')
    );
    const blob = new Blob([[headers.join(','), ...rowsCsv].join('\n')], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${SIH_PROBLEM.sihCode}_HydroMet_Dataset_${records.length}rows.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const tabs: { id: TabId; label: string; badge?: string; icon: React.ReactNode }[] = [
    {
      id: 'overview',
      label: 'Executive Analytics',
      icon: <LayoutDashboard className="w-4 h-4" />,
    },
    {
      id: 'dataset',
      label: 'Dataset & EDA Explorer',
      badge: records.length.toLocaleString(),
      icon: <Database className="w-4 h-4" />,
    },
    {
      id: 'models',
      label: '5-Model ML Studio',
      badge: '5 Models',
      icon: <Cpu className="w-4 h-4" />,
    },
    {
      id: 'math',
      label: 'Mathematical Engine',
      badge: '10×10 Matrix',
      icon: <Sigma className="w-4 h-4" />,
    },
    {
      id: 'simulator',
      label: 'Discharge & SHAP Simulator',
      icon: <Sliders className="w-4 h-4" />,
    },
  ];

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col">
      {/* Top Bar Contract: Maximum 3 distinct component groups separated by gap-8 */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="max-w-[1600px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          {/* Group 1: SIH1642 Problem Identity */}
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-lg bg-blue-600 text-white flex items-center justify-center shrink-0 shadow-xs">
              <Waves className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-xs">
                <span className="font-mono font-bold text-blue-700">{SIH_PROBLEM.sihCode}</span>
                <span className="text-slate-300">•</span>
                <span className="text-slate-500 truncate">
                  {SIH_PROBLEM.ministry} ({SIH_PROBLEM.nodalAgency})
                </span>
              </div>
              <h1 className="text-sm font-semibold text-slate-900 truncate">
                {SIH_PROBLEM.title}
              </h1>
            </div>
          </div>

          {/* Group 2: Primary Workspace Navigation */}
          <nav className="hidden xl:flex items-center gap-1">
            {tabs.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                    active
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <span className={active ? 'text-sky-400' : 'text-slate-400'}>{t.icon}</span>
                  <span>{t.label}</span>
                  {t.badge && (
                    <span
                      className={`text-[10px] font-mono tabular-nums ${
                        active ? 'text-slate-300' : 'text-slate-400'
                      }`}
                    >
                      ({t.badge})
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Group 3: Dataset Telemetry Readout & CSV Export */}
          <div className="flex items-center gap-4 shrink-0">
            <div className="hidden md:flex flex-col items-end text-xs">
              <span className="font-mono text-slate-600 tabular-nums">
                N = <strong className="text-slate-900">{analysis.nTotal.toLocaleString()}</strong> • 12 Basins • 10 Predictors
              </span>
              <span className="font-mono text-[11px] text-emerald-700 font-semibold tabular-nums">
                GBDT R² = {analysis.models.gbdt.r2.toFixed(4)} • NSE = {analysis.models.gbdt.nse.toFixed(4)}
              </span>
            </div>
            <button
              onClick={handleExportCsv}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 text-xs font-medium transition-colors cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-blue-600" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* Responsive Sub-navigation for smaller screens */}
        <div className="xl:hidden border-t border-slate-200 px-6 py-2 flex items-center gap-2 overflow-x-auto">
          {tabs.map((t) => {
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap cursor-pointer ${
                  active
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {t.icon}
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-6 py-6">
        {activeTab === 'overview' && (
          <OverviewTab
            problem={SIH_PROBLEM}
            analysis={analysis}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetExplorerTab
            problem={SIH_PROBLEM}
            records={records}
            analysis={analysis}
            onAddRecord={handleAddRecord}
            onAddBatchRecords={handleAddBatchRecords}
            onResetDataset={handleResetDataset}
            onExportCsv={handleExportCsv}
          />
        )}

        {activeTab === 'models' && (
          <ModelStudioTab
            problem={SIH_PROBLEM}
            analysis={analysis}
            hyper={hyper}
            onChangeHyper={setHyper}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
          />
        )}

        {activeTab === 'math' && (
          <MathFormulasTab
            problem={SIH_PROBLEM}
            analysis={analysis}
            hyper={hyper}
          />
        )}

        {activeTab === 'simulator' && (
          <SimulatorTab
            problem={SIH_PROBLEM}
            analysis={analysis}
          />
        )}
      </main>
    </div>
  );
}
