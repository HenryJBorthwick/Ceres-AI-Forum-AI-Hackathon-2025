import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function IpcChart({ data }) {
  // Prepare chart data: year, phases, isPredicted
  const chartData = data.map(d => ({
    year: d.year,
    phase1: d.ipc_phases.phase_1.percent_affected,
    phase2: d.ipc_phases.phase_2.percent_affected,
    phase3: d.ipc_phases.phase_3.percent_affected,
    phase4: d.ipc_phases.phase_4.percent_affected,
    phase5: d.ipc_phases.phase_5.percent_affected,
    isPredicted: d.is_predicted
  }));

  const getDashArray = isPredicted => isPredicted ? '3 3' : undefined;

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="year" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Legend />
        <Line dataKey="phase1" name="Phase 1 (Minimal)" stroke="#22c55e" strokeDasharray={getDashArray} connectNulls />
        <Line dataKey="phase2" name="Phase 2 (Stressed)" stroke="#eab308" strokeDasharray={getDashArray} connectNulls />
        <Line dataKey="phase3" name="Phase 3 (Crisis)" stroke="#f97316" strokeDasharray={getDashArray} connectNulls />
        <Line dataKey="phase4" name="Phase 4 (Emergency)" stroke="#ef4444" strokeDasharray={getDashArray} connectNulls />
        <Line dataKey="phase5" name="Phase 5 (Catastrophe)" stroke="#991b1b" strokeDasharray={getDashArray} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default IpcChart;
