import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';

function IpcChart({ data }) {
  // Use the chart-ready data from backend
  const chartData = data.map(d => d.chart_data);
  
  // Separate historical and predicted data for different line styling
  const historicalData = chartData.filter(d => !d.is_predicted);
  const predictedData = chartData.filter(d => d.is_predicted);
  
  // Create complete dataset with null values for predicted periods in historical data
  // and null values for historical periods in predicted data
  const completeData = chartData.map(d => ({
    year: d.year,
    // Historical data (null for predicted years)
    hist_phase1: !d.is_predicted ? d.phase1 : null,
    hist_phase2: !d.is_predicted ? d.phase2 : null,
    hist_phase3: !d.is_predicted ? d.phase3 : null,
    hist_phase4: !d.is_predicted ? d.phase4 : null,
    hist_phase5: !d.is_predicted ? d.phase5 : null,
    // Predicted data (null for historical years, but include connection point)
    pred_phase1: d.is_predicted ? d.phase1 : null,
    pred_phase2: d.is_predicted ? d.phase2 : null,
    pred_phase3: d.is_predicted ? d.phase3 : null,
    pred_phase4: d.is_predicted ? d.phase4 : null,
    pred_phase5: d.is_predicted ? d.phase5 : null,
    is_predicted: d.is_predicted
  }));
  
  // Add connection points for smooth transition
  if (historicalData.length > 0 && predictedData.length > 0) {
    const lastHistorical = historicalData[historicalData.length - 1];
    const connectionIndex = completeData.findIndex(d => d.year === lastHistorical.year);
    if (connectionIndex >= 0) {
      completeData[connectionIndex].pred_phase1 = lastHistorical.phase1;
      completeData[connectionIndex].pred_phase2 = lastHistorical.phase2;
      completeData[connectionIndex].pred_phase3 = lastHistorical.phase3;
      completeData[connectionIndex].pred_phase4 = lastHistorical.phase4;
      completeData[connectionIndex].pred_phase5 = lastHistorical.phase5;
    }
  }

  // Check if this specific graph has both historical and predicted data
  const hasHistoricalData = historicalData.length > 0;
  const hasPredictedData = predictedData.length > 0;
  const showGuide = hasHistoricalData && hasPredictedData;

  return (
    <div>
      {/* Name of the graph */}
      <h2 className="text-xl font-bold text-center mb-4 text-gray-800">
        Ceres IPC Pulse
      </h2>
      {showGuide && (
        <p className="text-sm text-gray-600 mb-2 text-center">
          <span className="font-medium">Chart Guide:</span> Solid lines = Historical data, Dashed lines = Predicted data
        </p>
      )}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={completeData} margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="year">
          <Label value="Year" position="insideBottom" offset={-5} style={{ textAnchor: 'middle' }} />
        </XAxis>
        <YAxis domain={[0, 100]}>
          <Label value="Population Affected (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
        </YAxis>
        <Tooltip 
          content={(props) => {
            const { active, payload, label } = props;
            if (!active || !payload || !payload.length) return null;

            // Get the data for this point
            const data = payload[0].payload;
            const isPredicted = data.is_predicted;

            // Use hist_ for historical, pred_ for predicted
            const phasePrefix = isPredicted ? 'pred_phase' : 'hist_phase';
            const title = `${label} (${isPredicted ? 'Predicted' : 'Historical'})`;

            return (
              <div className="bg-white p-4 border rounded shadow">
                <p className="font-bold">{title}</p>
                <ul>
                  <li style={{color: '#22c55e'}}>Phase 1: {data[`${phasePrefix}1`]}%</li>
                  <li style={{color: '#eab308'}}>Phase 2: {data[`${phasePrefix}2`]}%</li>
                  <li style={{color: '#f97316'}}>Phase 3: {data[`${phasePrefix}3`]}%</li>
                  <li style={{color: '#ef4444'}}>Phase 4: {data[`${phasePrefix}4`]}%</li>
                  <li style={{color: '#991b1b'}}>Phase 5: {data[`${phasePrefix}5`]}%</li>
                </ul>
              </div>
            );
          }}
        />
        {/* Legend below the x-axis */}
        <Legend wrapperStyle={{ paddingTop: '20px' }} />
        
        {/* Historical data lines - solid */}
        <Line dataKey="hist_phase1" name="Phase 1 (Minimal)" stroke="#22c55e" strokeWidth={2} connectNulls />
        <Line dataKey="hist_phase2" name="Phase 2 (Stressed)" stroke="#eab308" strokeWidth={2} connectNulls />
        <Line dataKey="hist_phase3" name="Phase 3 (Crisis)" stroke="#f97316" strokeWidth={2} connectNulls />
        <Line dataKey="hist_phase4" name="Phase 4 (Emergency)" stroke="#ef4444" strokeWidth={2} connectNulls />
        <Line dataKey="hist_phase5" name="Phase 5 (Catastrophe)" stroke="#991b1b" strokeWidth={2} connectNulls />
        
        {/* Predicted data lines - dashed with reduced opacity, hidden from legend */}
        <Line dataKey="pred_phase1" name="" stroke="#22c55e" strokeWidth={2} strokeDasharray="5 5" strokeOpacity={0.7} connectNulls legendType="none" />
        <Line dataKey="pred_phase2" name="" stroke="#eab308" strokeWidth={2} strokeDasharray="5 5" strokeOpacity={0.7} connectNulls legendType="none" />
        <Line dataKey="pred_phase3" name="" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" strokeOpacity={0.7} connectNulls legendType="none" />
        <Line dataKey="pred_phase4" name="" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" strokeOpacity={0.7} connectNulls legendType="none" />
        <Line dataKey="pred_phase5" name="" stroke="#991b1b" strokeWidth={2} strokeDasharray="5 5" strokeOpacity={0.7} connectNulls legendType="none" />
      </LineChart>
    </ResponsiveContainer>
    </div>
  );
}

export default IpcChart;
