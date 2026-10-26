export type FeatureKey = 'x1' | 'x2' | 'x3' | 'x4' | 'x5' | 'x6' | 'x7' | 'x8' | 'x9' | 'x10';

export const FEATURE_KEYS: FeatureKey[] = [
  'x1',
  'x2',
  'x3',
  'x4',
  'x5',
  'x6',
  'x7',
  'x8',
  'x9',
  'x10',
];

export interface FeatureSpec {
  key: FeatureKey;
  code: string;
  label: string;
  symbol: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  defaultVal: number;
}

export interface DataRecord {
  id: string;
  timestamp: string;
  station: string;
  seasonOrZone: string;
  x1: number; // 24h Basin Rainfall (mm)
  x2: number; // 7-Day Antecedent Precip Index API_7 (mm)
  x3: number; // Upstream Gauge Discharge Q_up (m³/s)
  x4: number; // Soil Volumetric Saturation θ_soil (%)
  x5: number; // Reservoir Live Storage V_res (%)
  x6: number; // Catchment Slope S_c (deg)
  x7: number; // Evapotranspiration ET_0 (mm/day)
  x8: number; // SCS Runoff Curve Number CN (52-96)
  x9: number; // Basin Lag Time T_lag (hrs)
  x10: number; // Spillway Gate Release Q_spill (m³/s)
  waterLevelM: number; // Gauge Water Level (m)
  target: number; // Peak Flood Discharge Q_peak (m³/s)
  riskClass: 'Normal' | 'Advisory' | 'Severe';
  isCritical: number;
}

export interface SIHProblemConfig {
  sihCode: string;
  ministry: string;
  nodalAgency: string;
  title: string;
  physicalEquationLatex: string;
  targetLabel: string;
  targetSymbol: string;
  targetUnit: string;
  criticalThreshold: number;
  advisoryThreshold: number;
  criticalClassLabel: string;
  advisoryClassLabel: string;
  normalClassLabel: string;
  stations: string[];
  features: FeatureSpec[];
  records: DataRecord[];
}

function createSeededRandom(seed: number) {
  let a = seed;
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rand: () => number): number {
  const u1 = Math.max(1e-7, rand());
  const u2 = rand();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function generateCWC2500Records(count = 2500): DataRecord[] {
  const rand = createSeededRandom(16422500);
  const stations = [
    'Brahmaputra - Pandu',
    'Godavari - Polavaram',
    'Mahanadi - Hirakud',
    'Krishna - Srisailam',
    'Narmada - Sardar Sarovar',
    'Ganga - Farakka',
    'Yamuna - Hathnikund',
    'Kaveri - Mettur',
    'Tapi - Ukai Dam',
    'Damodar - Maithon',
    'Sutlej - Bhakra Nangal',
    'Kosi - Birpur Barrage',
  ];
  const seasons = [
    'SW Monsoon Peak',
    'Active Monsoon Trough',
    'Cyclonic Depression',
    'Early Monsoon Onset',
    'Retreating Monsoon',
  ];

  const stationBaseFlow: Record<string, number> = {
    'Brahmaputra - Pandu': 6100,
    'Godavari - Polavaram': 4500,
    'Mahanadi - Hirakud': 3750,
    'Krishna - Srisailam': 3200,
    'Narmada - Sardar Sarovar': 3550,
    'Ganga - Farakka': 5300,
    'Yamuna - Hathnikund': 2500,
    'Kaveri - Mettur': 2050,
    'Tapi - Ukai Dam': 2850,
    'Damodar - Maithon': 2300,
    'Sutlej - Bhakra Nangal': 2950,
    'Kosi - Birpur Barrage': 4800,
  };

  const records: DataRecord[] = [];
  const startTime = new Date('2024-06-01T00:00:00Z').getTime();

  for (let i = 0; i < count; i++) {
    const station = stations[i % stations.length];
    const season = seasons[Math.floor((i / 12) % seasons.length)];
    const monsoonIntensity =
      season === 'SW Monsoon Peak'
        ? 1.32
        : season === 'Cyclonic Depression'
          ? 1.24
          : season === 'Active Monsoon Trough'
            ? 1.12
            : season === 'Retreating Monsoon'
              ? 0.85
              : 0.72;

    // x1: 24h Basin Rainfall (mm, 6 - 350)
    const x1 = Number(
      Math.max(6, Math.min(350, (38 + rand() * 198) * monsoonIntensity + gaussianRandom(rand) * 14)).toFixed(1)
    );
    // x2: 7-Day Antecedent Precip Index API_7 (mm, 18 - 495)
    const x2 = Number(
      Math.max(18, Math.min(495, x1 * (1.1 + rand() * 0.5) + 26 * monsoonIntensity + gaussianRandom(rand) * 16)).toFixed(1)
    );
    // x3: Upstream Gauge Discharge Q_up (m³/s, 280 - 15200)
    const baseQ = stationBaseFlow[station] || 3200;
    const x3 = Number(
      Math.max(280, Math.min(15200, baseQ * (0.52 + 0.68 * monsoonIntensity) + x1 * 11.2 + gaussianRandom(rand) * 310)).toFixed(0)
    );
    // x4: Soil Volumetric Saturation θ_soil (%, 18 - 99)
    const x4 = Number(
      Math.max(18, Math.min(99, 32 + 0.14 * x2 + 8 * monsoonIntensity + gaussianRandom(rand) * 5.0)).toFixed(1)
    );
    // x5: Reservoir Live Storage V_res (%, 24 - 99.5)
    const x5 = Number(
      Math.max(24, Math.min(99.5, 44 + 0.082 * x2 + 0.0021 * x3 + gaussianRandom(rand) * 6.0)).toFixed(1)
    );
    // x6: Catchment Terrain Slope S_c (deg, 1.2 - 19.5)
    const slopeBase =
      station.includes('Brahmaputra') || station.includes('Sutlej') || station.includes('Kosi')
        ? 12.6
        : station.includes('Narmada') || station.includes('Mahanadi')
          ? 7.8
          : 4.9;
    const x6 = Number(Math.max(1.2, Math.min(19.5, slopeBase + (rand() - 0.5) * 4.4)).toFixed(2));
    // x7: Evapotranspiration Loss ET_0 (mm/day, 1.5 - 9.8)
    const x7 = Number(
      Math.max(1.5, Math.min(9.8, 7.8 - 0.012 * x1 - 0.025 * x4 + rand() * 2.2 + gaussianRandom(rand) * 0.4)).toFixed(2)
    );
    // x8: SCS Runoff Curve Number CN (52 - 96)
    const x8 = Number(
      Math.max(52, Math.min(96, 61 + 0.26 * x4 + 0.45 * x6 + gaussianRandom(rand) * 3.2)).toFixed(1)
    );
    // x9: Basin Concentration Lag Time T_lag (hrs, 3.5 - 38.0)
    const x9 = Number(
      Math.max(3.5, Math.min(38.0, 29.5 - 0.92 * x6 - 0.045 * x4 + rand() * 5.5)).toFixed(1)
    );
    // x10: Spillway Gate Outflow Release Q_spill (m³/s, 100 - 8800)
    const x10 = Number(
      Math.max(
        100,
        Math.min(8800, (x5 > 78 ? (x5 - 70) * 115 : 220) + 0.34 * x3 + 5.8 * x1 + gaussianRandom(rand) * 180)
      ).toFixed(0)
    );

    // Physical Coupled Muskingum + SCS-CN + Spillway Peak Discharge
    const scsRetention = Math.max(5, 25400 / x8 - 254);
    const scsRunoff = x1 > 0.2 * scsRetention ? Math.pow(x1 - 0.2 * scsRetention, 2) / (x1 + 0.8 * scsRetention) : 0;

    const target = Number(
      Math.max(
        480,
        0.66 * x3 +
          0.38 * x10 +
          11.4 * x1 +
          4.8 * x2 +
          16.5 * x4 +
          15.2 * x5 +
          58.0 * x6 -
          38.0 * x7 +
          19.5 * x8 -
          24.0 * x9 +
          8.5 * scsRunoff +
          gaussianRandom(rand) * 195
      ).toFixed(0)
    );

    const waterLevelM = Number((82.5 + Math.pow(target / 180, 0.68)).toFixed(2));

    const riskClass: 'Normal' | 'Advisory' | 'Severe' =
      target >= 10400 ? 'Severe' : target >= 7700 ? 'Advisory' : 'Normal';

    const obsDate = new Date(startTime + i * 2 * 3600 * 1000);
    const timestamp = obsDate.toISOString().slice(0, 16).replace('T', ' ');

    records.push({
      id: `CWC-${String(i + 1).padStart(4, '0')}`,
      timestamp,
      station,
      seasonOrZone: season,
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

  return records;
}

export const SIH_PROBLEM: SIHProblemConfig = {
  sihCode: 'SIH1642',
  ministry: 'Ministry of Jal Shakti',
  nodalAgency: 'Central Water Commission (CWC) · National Institute of Hydrology',
  title: 'Spatio-Temporal River Basin Flood Discharge Forecasting & Reservoir Inflow Optimization',
  physicalEquationLatex:
    'Q_{t+1} = C_0 I_{t+1} + C_1 I_t + C_2 Q_t + Q_{\\text{spill}} + \\frac{(R_{24} - 0.2S_{\\text{CN}})^2}{R_{24} + 0.8S_{\\text{CN}}}',
  targetLabel: 'Peak Flood Discharge',
  targetSymbol: 'Q_peak',
  targetUnit: 'm³/s',
  criticalThreshold: 10400,
  advisoryThreshold: 7700,
  criticalClassLabel: 'Severe Flood Warning',
  advisoryClassLabel: 'Flood Watch Advisory',
  normalClassLabel: 'Nominal Basin Flow',
  stations: [
    'Brahmaputra - Pandu',
    'Godavari - Polavaram',
    'Mahanadi - Hirakud',
    'Krishna - Srisailam',
    'Narmada - Sardar Sarovar',
    'Ganga - Farakka',
    'Yamuna - Hathnikund',
    'Kaveri - Mettur',
    'Tapi - Ukai Dam',
    'Damodar - Maithon',
    'Sutlej - Bhakra Nangal',
    'Kosi - Birpur Barrage',
  ],
  features: [
    {
      key: 'x1',
      code: 'rainfall_24h_mm',
      label: '24h Basin Rainfall',
      symbol: 'R_24',
      unit: 'mm',
      min: 6,
      max: 350,
      step: 2,
      defaultVal: 170,
    },
    {
      key: 'x2',
      code: 'api_7day_mm',
      label: '7-Day Antecedent Precip',
      symbol: 'API_7',
      unit: 'mm',
      min: 18,
      max: 495,
      step: 5,
      defaultVal: 240,
    },
    {
      key: 'x3',
      code: 'upstream_flow_cumecs',
      label: 'Upstream Gauge Flow',
      symbol: 'Q_up',
      unit: 'm³/s',
      min: 280,
      max: 15200,
      step: 100,
      defaultVal: 5600,
    },
    {
      key: 'x4',
      code: 'soil_moisture_pct',
      label: 'Soil Saturation',
      symbol: 'θ_soil',
      unit: '%',
      min: 18,
      max: 99,
      step: 1,
      defaultVal: 76,
    },
    {
      key: 'x5',
      code: 'reservoir_storage_pct',
      label: 'Reservoir Storage Fill',
      symbol: 'V_res',
      unit: '%',
      min: 24,
      max: 99.5,
      step: 1,
      defaultVal: 84,
    },
    {
      key: 'x6',
      code: 'catchment_slope_deg',
      label: 'Catchment Slope',
      symbol: 'S_c',
      unit: 'deg',
      min: 1.2,
      max: 19.5,
      step: 0.2,
      defaultVal: 8.8,
    },
    {
      key: 'x7',
      code: 'evapotranspiration_mm',
      label: 'Evapotranspiration',
      symbol: 'ET_0',
      unit: 'mm/d',
      min: 1.5,
      max: 9.8,
      step: 0.2,
      defaultVal: 4.2,
    },
    {
      key: 'x8',
      code: 'scs_curve_number',
      label: 'SCS Curve Number',
      symbol: 'CN',
      unit: 'idx',
      min: 52,
      max: 96,
      step: 1,
      defaultVal: 81,
    },
    {
      key: 'x9',
      code: 'basin_lag_time_hrs',
      label: 'Basin Lag Time',
      symbol: 'T_lag',
      unit: 'hrs',
      min: 3.5,
      max: 38.0,
      step: 0.5,
      defaultVal: 14.5,
    },
    {
      key: 'x10',
      code: 'spillway_outflow_cumecs',
      label: 'Spillway Gate Outflow',
      symbol: 'Q_spill',
      unit: 'm³/s',
      min: 100,
      max: 8800,
      step: 50,
      defaultVal: 2900,
    },
  ],
  records: generateCWC2500Records(2500),
};
